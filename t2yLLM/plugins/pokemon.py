from .pluginManager import APIBase, logger
from pathlib import Path
from rapidfuzz import process, fuzz
from difflib import SequenceMatcher
import subprocess
import requests
import re


class PokeAPI(APIBase):
    @classmethod
    def init(cls, **kwargs):
        return cls(**kwargs)

    def __init__(self, **kwargs):
        self.config = kwargs.get("config")
        self.nlp = kwargs.get("nlp")
        self.language = kwargs.get("language")
        self.pokemon_list = []
        self.query = False
        self.pokemon_name = ""
        self.pokejson = None
        self.pokemon_phonetics = []
        self.activate_memory = True
        self.silent_execution = False

        try:
            package_dir = Path(__file__).resolve().parents[1] / "config"
            pokemon_list_name = f"pokemon_list_{self.config.general.lang}"
            pokemon_file = package_dir / getattr(self.config.pokemon, pokemon_list_name)
            with open(pokemon_file, "r", encoding="utf-8") as file:
                self.pokemon_list = [line.strip() for line in file if line.strip()]
            try:
                phonetics_file = package_dir / self.config.pokemon.pokemon_phonetics
                with open(phonetics_file, "r", encoding="utf-8") as file:
                    self.pokemon_phonetics = [
                        line.strip() for line in file if line.strip()
                    ]
            except Exception:
                logger.warning("Could not load Pokemon phonetics file")

        except Exception as e:
            logger.error(f"Error loading pokemon file: {e}")

    @property
    def name(self) -> str:
        return "PokeAPI"

    @property
    def filename(self) -> str:
        return "pokemon"

    @property
    def is_enabled(self) -> bool:
        if self.name() or self.filename() in self.config.plugins.enabled_plugins:
            return True
        else:
            return False

    @property
    def memory(self) -> bool:
        return self.activate_memory

    def is_query(self, user_input, threshold=0.91) -> bool:
        if not self.pokemon_list:
            return False
        self.query = False
        self.pokemon_name = ""
        try:
            threshold = self.config.pokemon.pokemon_find_threshold
        except Exception:
            threshold = 0.91
        words = user_input.split()
        accent_map = str.maketrans("éèêë", "eeee")

        for word in words:
            word = word.translate(accent_map)
            word = word.capitalize()
            matches = process.extract(
                word,
                self.pokemon_list,
                scorer=fuzz.ratio,
                limit=5,
            )
            logger.debug(f"Matches for '{word}': {matches}")

            best_score = 0

            if self.pokemon_phonetics:
                _, best_score = self.detect_pokephonetics(word)
            if (
                matches
                and (matches[0][1] >= threshold)
                or (
                    (matches[0][1] >= (threshold - 10))
                    and (best_score >= threshold + 4)
                )
            ):
                self.query = True
                self.pokemon_name = matches[0][0]
                logger.info(
                    f"Pokemon detected: {self.pokemon_name} (score: {matches[0][1]})"
                )
                return True

        # self.query = False
        return False

    def get_ipa_fr(self, word):
        try:
            result = subprocess.run(
                [
                    "sudo",
                    "-u",
                    self.config.general.unprivileged_user,
                    "espeak-ng",
                    "-v",
                    self.config.general.lang,
                    "--ipa",
                    "-q",
                    word,
                ],
                capture_output=True,
                text=True,
            )
            phonetic = re.sub(r"\([a-z]{2,3}\)", "", result.stdout)
            return phonetic.strip()
        except Exception as e:
            logger.warning(f"Error getting IPA: {e}")

    def similarity(self, a, b):
        return round(SequenceMatcher(None, a, b).ratio() * 100, 2)

    def detect_pokephonetics(self, user_input):
        """
        it is not working really well because there isnt really
        something worknig in french (but plenty in english)
        lets say it is not that accurate
        """
        doc = self.nlp(user_input)
        words = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]]
        # more special tokens exist in english but french model is limited

        best_match_index = None
        best_score = 0
        for word in words:
            word_ipa = self.get_ipa_fr(word)

            word_scores = []
            for i, pokemon_ipa in enumerate(self.pokemon_phonetics):
                score = self.similarity(word_ipa, pokemon_ipa)
                word_scores.append((i, score))

            if word_scores:
                word_scores.sort(key=lambda x: x[1], reverse=True)
                index, score = word_scores[0]

                if score > best_score:
                    best_match_index = index
                    best_score = score

        if best_match_index is None:
            return (None, 0)

        return best_match_index, best_score

    def search(self, user_input=None, **kwargs):
        if not self.pokemon_name:
            result = {"success": False, "error": "No Pokemon name detected"}
        if self.config.general.lang != "fr":
            result = self.pokeapi(self.pokemon_name)
        else:
            result = self.tyradex(self.pokemon_name)
        return result

    def tyradex(self, pokemon_name):
        try:
            url = f"https://tyradex.vercel.app/api/v1/pokemon/{pokemon_name.lower()}"
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
            }
            answer = requests.get(url, headers=headers, timeout=5.0)
            if answer.status_code != 200:
                return {
                    "success": False,
                    "error": f"Aucun résultat Tyradex (statut: {answer.status_code})",
                }
            self.pokejson = answer.json()

            if "name" in self.pokejson and "data" not in self.pokejson:
                self.pokejson = {"data": self.pokejson}

            return {"success": True, "data": self.pokejson}

        except Exception as e:
            return {
                "success": False,
                "error": f"Erreur lors de la recherche Tyradex: {str(e)}",
            }

    def format(self) -> str:
        """
        translates JSON from Tyradex or PokeAPI into
        text that the LLM can process as context
        """
        if not self.pokejson:
            return "Aucune donnée disponible pour ce Pokémon."

        if self.config.general.lang != "fr":
            return self.pokeapi_translate(
                self.pokejson,
                lang=self.config.general.lang
                if self.config.general.lang in ["en", "es", "de", "it"]
                else "en",
            )
        else:
            if "data" in self.pokejson:
                json_data = self.pokejson["data"]
            else:
                json_data = self.pokejson

            if not json_data:
                return "Aucune donnée disponible pour ce Pokémon."
            else:
                try:
                    name_fr = json_data.get("name", {}).get("fr", "Inconnu")
                    pokedex_id = json_data.get("pokedex_id", "Inconnu")
                    category = json_data.get("category", "Inconnu")
                    generation = json_data.get("generation", "Inconnu")

                    description = f"{name_fr} est un Pokémon de type "

                    types = json_data.get("types", [])
                    type_names = [type_info.get("name", "") for type_info in types]
                    if len(type_names) == 1:
                        description += f"{type_names[0]}"
                    elif len(type_names) == 2:
                        description += f"{type_names[0]} et {type_names[1]}"
                    else:
                        description += (
                            ", ".join(type_names[:-1]) + f" et {type_names[-1]}"
                        )

                    description += f". Il porte le numéro {pokedex_id} du Pokédex"
                    if generation:
                        description += f" et est apparu dans la génération {generation}"
                    description += "."

                    if category and category != "Inconnu":
                        description += f" Il est connu comme le {category}."

                    try:
                        height = json_data.get("height", "Inconnu")
                        weight = json_data.get("weight", "Inconnu")
                        description += f" Il mesure {height} et pèse {weight}."
                    except (AttributeError, NameError):
                        pass

                    try:
                        talents = json_data.get("talents", []) or []
                        if talents:
                            talent_names = []
                            tc_talents = []

                            for talent in talents:
                                name = talent.get("name", "")
                                is_tc = talent.get("tc", False)

                                if is_tc:
                                    tc_talents.append(name)
                                else:
                                    talent_names.append(name)

                            if talent_names:
                                description += (
                                    f" Ses talents sont : {', '.join(talent_names)}"
                                )

                                if tc_talents:
                                    description += f" et il possède le talent caché : {', '.join(tc_talents)}"
                                description += "."
                            elif tc_talents:
                                description += f" Il possède uniquement le talent caché : {', '.join(tc_talents)}."
                    except (AttributeError, NameError):
                        pass

                    try:
                        stats = json_data.get("stats", {}) or {}
                        if stats:
                            description += f" Ses statistiques de base sont : {stats.get('hp', 0)} PV, "
                            description += f"{stats.get('atk', 0)} en Attaque, {stats.get('def', 0)} en Défense, "
                            description += f"{stats.get('spe_atk', 0)} en Attaque Spéciale, {stats.get('spe_def', 0)} en Défense Spéciale "
                            description += f"et {stats.get('vit', 0)} en Vitesse."
                    except (AttributeError, NameError):
                        pass

                    try:
                        evolution = json_data.get("evolution", {})

                        if evolution:
                            try:
                                pre_evolutions = evolution.get("pre", []) or []
                                if pre_evolutions and len(pre_evolutions) > 0:
                                    pre_evo = pre_evolutions[0]
                                    pre_name = pre_evo.get("name", "")
                                    pre_condition = pre_evo.get("condition", "")

                                    if pre_name and pre_condition:
                                        description += f" {name_fr} est l'évolution de {pre_name} ({pre_condition})."
                            except (AttributeError, NameError, IndexError):
                                pass

                            try:
                                next_evolutions = evolution.get("next", []) or []
                                if next_evolutions and len(next_evolutions) > 0:
                                    next_evo = next_evolutions[0]
                                    next_name = next_evo.get("name", "")
                                    next_condition = next_evo.get("condition", "")

                                    if next_name and next_condition:
                                        description += f" Il évolue en {next_name} ({next_condition})."
                            except (AttributeError, NameError, IndexError):
                                pass

                            try:
                                mega = evolution.get("mega", None)
                                if mega:
                                    description += " Il possède une méga-évolution."
                            except (AttributeError, NameError):
                                pass
                    except AttributeError:
                        pass

                    try:
                        catch_rate = json_data.get("catch_rate", None)
                        if catch_rate is not None:
                            description += f" Son taux de capture est de {catch_rate}."
                    except (AttributeError, NameError):
                        pass

                    try:
                        sexe = json_data.get("sexe", {}) or {}
                        male_rate = sexe.get("male", 0)
                        female_rate = sexe.get("female", 0)

                        if male_rate > 0 and female_rate > 0:
                            description += f" La répartition des sexes est de {male_rate}% de mâles et {female_rate}% de femelles."
                        elif male_rate == 0 and female_rate == 0:
                            description += " Ce Pokémon est asexué."
                        elif male_rate == 0:
                            description += " Ce Pokémon est exclusivement femelle."
                        elif female_rate == 0:
                            description += " Ce Pokémon est exclusivement mâle."
                    except (AttributeError, NameError):
                        pass

                    try:
                        egg_groups = json_data.get("egg_groups", [])
                        if egg_groups:
                            if len(egg_groups) == 1:
                                description += (
                                    f" Il appartient au groupe d'œuf {egg_groups[0]}."
                                )
                            else:
                                description += f" Il appartient aux groupes d'œufs {' et '.join(egg_groups)}."
                    except (AttributeError, NameError):
                        pass

                    try:
                        resistances = json_data.get("resistances", [])

                        weaknesses = []
                        strengths = []
                        immunities = []

                        for res in resistances:
                            type_name = res.get("name", "")
                            multiplier = res.get("multiplier", 1)

                            if multiplier > 1:
                                weaknesses.append(f"{type_name} (x{multiplier})")
                            elif multiplier < 1 and multiplier > 0:
                                strengths.append(f"{type_name} (x{multiplier})")
                            elif multiplier == 0:
                                immunities.append(type_name)

                        if weaknesses:
                            description += f" Il est faible contre les attaques de type {', '.join(weaknesses)}."

                        if strengths:
                            description += f" Il résiste aux attaques de type {', '.join(strengths)}."

                        if immunities:
                            description += f" Il est immunisé contre les attaques de type {', '.join(immunities)}."
                    except (AttributeError, NameError):
                        pass

                    try:
                        formes = json_data.get("formes", None)
                        if formes:
                            description += " Ce Pokémon possède différentes formes."
                    except (AttributeError, NameError):
                        pass

                except (AttributeError, KeyError, IndexError, ValueError, TypeError):
                    description = "info non trouvée sur ce pokémon"
                    self.wiki_query = True

            return description

    def pokeapi(self, pokemon_name):
        try:
            pokemon_name = pokemon_name.lower().strip()
            url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name}"
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
            }
            answer = requests.get(url, headers=headers, timeout=10)

            if answer.status_code != 200:
                return {
                    "success": False,
                    "error": f"Pokemon not found on PokeAPI (status: {answer.status_code})",
                }

            data = answer.json()

            species_url = data.get("species", {}).get("url", "")
            species_data = {}
            if species_url:
                species_response = requests.get(
                    species_url, headers=headers, timeout=5.0
                )
                if species_response.status_code == 200:
                    species_data = species_response.json()
            data = self.pokeapi_translate(data, species_data)
            self.pokejson = data

            return {"success": True, "data": data}

        except Exception as e:
            return {
                "success": False,
                "error": f"Error in PokeAPI search: {str(e)}",
            }

    @staticmethod
    def pokeapi2tyradex(pokemon_data, species_data):
        names = {"en": pokemon_data.get("name", "Unknown").title()}
        if species_data and "names" in species_data:
            for name_entry in species_data["names"]:
                lang = name_entry.get("language", {}).get("name", "")
                if lang == "fr":
                    names["fr"] = name_entry.get("name", names["en"])
                elif lang == "en":
                    names["en"] = name_entry.get("name", names["en"])

        if "fr" not in names:
            names["fr"] = names["en"]

        types = []
        for type_slot in pokemon_data.get("types", []):
            type_name = type_slot.get("type", {}).get("name", "")
            if type_name:
                types.append({"name": type_name.capitalize()})

        abilities = []
        for ability_slot in pokemon_data.get("abilities", []):
            ability_info = ability_slot.get("ability", {})
            is_hidden = ability_slot.get("is_hidden", False)
            abilities.append(
                {
                    "name": ability_info.get("name", "").replace("-", " ").title(),
                    "tc": is_hidden,  # tc = talent caché (hidden ability)
                }
            )

        stats = {}
        stat_mapping = {
            "hp": "hp",
            "attack": "atk",
            "defense": "def",
            "special-attack": "spe_atk",
            "special-defense": "spe_def",
            "speed": "vit",
        }

        for stat in pokemon_data.get("stats", []):
            stat_name = stat.get("stat", {}).get("name", "")
            if stat_name in stat_mapping:
                stats[stat_mapping[stat_name]] = stat.get("base_stat", 0)

        generation_url = species_data.get("generation", {}).get("url", "")
        generation = 1
        if generation_url:
            try:
                generation = int(generation_url.rstrip("/").split("/")[-1])
            except Exception:
                generation = 1

        category = "Pokémon"
        if species_data and "genera" in species_data:
            for genus in species_data["genera"]:
                if genus.get("language", {}).get("name") == "fr":
                    category = genus.get("genus", "Pokémon")
                    break
                elif (
                    genus.get("language", {}).get("name") == "en"
                    and category == "Pokémon"
                ):
                    category = genus.get("genus", "Pokémon")

        transformed = {
            "name": names,
            "pokedex_id": pokemon_data.get("id", 0),
            "generation": generation,
            "category": category,
            "types": types,
            "talents": abilities,
            "stats": stats,
            "height": f"{pokemon_data.get('height', 0) / 10} m",
            "weight": f"{pokemon_data.get('weight', 0) / 10} kg",
            "catch_rate": species_data.get("capture_rate", None),
            "base_experience": pokemon_data.get("base_experience", None),
        }

        gender_rate = species_data.get("gender_rate", -1)
        if gender_rate >= 0:
            female_rate = (gender_rate / 8) * 100
            male_rate = 100 - female_rate
            transformed["sexe"] = {"male": male_rate, "female": female_rate}
        else:
            transformed["sexe"] = {"male": 0, "female": 0}

        egg_groups = []
        for group in species_data.get("egg_groups", []):
            group_name = group.get("name", "")
            if group_name:
                egg_groups.append(group_name.replace("-", " ").title())
        if egg_groups:
            transformed["egg_groups"] = egg_groups

        transformed["evolution"] = {"pre": [], "next": [], "mega": None}

        transformed["resistances"] = []

        return {"data": transformed}

    @staticmethod
    def pokeapi_translate(pokejson, lang="en"):
        json_data = pokejson
        if not json_data or "data" not in json_data:
            return "No data available for this Pokémon."

        try:
            data = json_data["data"]

            name = data.get("name", {}).get(
                lang, data.get("name", {}).get("en", "Unknown")
            )
            pokedex_id = data.get("pokedex_id", "Unknown")
            category = data.get("category", "Unknown")
            generation = data.get("generation", "Unknown")

            description = f"{name} is a "

            types = data.get("types", [])
            type_names = [type_info.get("name", "") for type_info in types]
            if len(type_names) == 1:
                description += f"{type_names[0]}-type Pokémon"
            elif len(type_names) == 2:
                description += f"{type_names[0]}/{type_names[1]}-type Pokémon"
            else:
                description += "Pokémon"

            description += f". It is number {pokedex_id} in the Pokédex"
            if generation and generation != "Unknown":
                description += f" and first appeared in Generation {generation}"
            description += "."

            if category and category != "Unknown" and category != "Pokémon":
                description += f" It is known as the {category}."

            height = data.get("height", "Unknown")
            weight = data.get("weight", "Unknown")
            if height != "Unknown" and weight != "Unknown":
                description += f" It is {height} tall and weighs {weight}."

            abilities = data.get("talents", []) or []
            if abilities:
                regular_abilities = []
                hidden_abilities = []

                for ability in abilities:
                    name = ability.get("name", "")
                    is_hidden = ability.get("tc", False)

                    if is_hidden:
                        hidden_abilities.append(name)
                    else:
                        regular_abilities.append(name)

                if regular_abilities:
                    description += f" Its abilities are: {', '.join(regular_abilities)}"

                    if hidden_abilities:
                        description += f", and it has the hidden ability: {
                            ', '.join(hidden_abilities)
                        }"
                    description += "."
                elif hidden_abilities:
                    description += f" It only has the hidden ability: {
                        ', '.join(hidden_abilities)
                    }."

            stats = data.get("stats", {}) or {}
            if stats:
                description += f" Its base stats are: {stats.get('hp', 0)} HP, "
                description += (
                    f"{stats.get('atk', 0)} Attack, {stats.get('def', 0)} Defense, "
                )
                description += f"{stats.get('spe_atk', 0)} Special Attack, {
                    stats.get('spe_def', 0)
                } Special Defense, "
                description += f"and {stats.get('vit', 0)} Speed."

            catch_rate = data.get("catch_rate", None)
            if catch_rate is not None:
                description += f" Its catch rate is {catch_rate}."

            sexe = data.get("sexe", {}) or {}
            male_rate = sexe.get("male", 0)
            female_rate = sexe.get("female", 0)

            if male_rate > 0 and female_rate > 0:
                description += f" The gender distribution is {male_rate}% male and {
                    female_rate
                }% female."
            elif male_rate == 0 and female_rate == 0:
                description += " This Pokémon is genderless."
            elif male_rate == 0:
                description += " This Pokémon is exclusively female."
            elif female_rate == 0:
                description += " This Pokémon is exclusively male."

            egg_groups = data.get("egg_groups", [])
            if egg_groups:
                if len(egg_groups) == 1:
                    description += f" It belongs to the {egg_groups[0]} egg group."
                else:
                    description += (
                        f" It belongs to the {' and '.join(egg_groups)} egg groups."
                    )

        except Exception as e:
            logger.error(f"Error translating PokeAPI info: {str(e)}")
            description = "Information not found for this Pokémon."

        return description

    def search_terms(self, user_input):
        if self.pokemon_name:
            return f"pokemon {self.pokemon_name}"
        return ""
