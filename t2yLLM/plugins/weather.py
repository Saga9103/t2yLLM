from .pluginManager import APIBase, logger
import os
import requests
import time
from datetime import datetime
from collections import Counter


class WeatherAPI(APIBase):
    @classmethod
    def init(cls, **kwargs):
        return cls(**kwargs)

    def __init__(self, **kwargs):
        self.config = kwargs.get("config")
        self.nlp = kwargs.get("nlp")
        self.language = kwargs.get("language")
        self.query = False
        self.activate_memory = False
        self.location_name = None
        self.forecast_days = 0
        self.is_weekend_query = False
        self.weather_data = None
        self.forecast_data = None
        self.silent_execution = False

    @property
    def name(self) -> str:
        return "WeatherAPI"

    @property
    def filename(self) -> str:
        return "weather"

    @property
    def is_enabled(self) -> bool:
        if self.name or self.filename in self.config.plugins.enabled_plugins:
            return True
        else:
            return False

    @property
    def memory(self):
        return self.activate_memory

    def is_query(self, user_input) -> bool:
        user_input_lower = user_input.lower()

        weather_keywords = self.language.weather_keywords
        temporal_indicators = self.language.temporal_indicators
        condition_patterns = self.language.condition_patterns
        forecast_patterns = self.language.forecast_patterns

        is_weather = any(keyword in user_input_lower for keyword in weather_keywords)
        is_temporal = any(
            indicator in user_input_lower for indicator in temporal_indicators
        )
        is_condition = any(
            pattern in user_input_lower for pattern in condition_patterns
        )
        is_forecast = any(pattern in user_input_lower for pattern in forecast_patterns)

        self.query = is_weather or is_temporal or is_condition or is_forecast

        if self.query:
            self.forecast_days = self.get_forecast_days(user_input_lower)
            self.location_name = self.get_location_from_input(user_input)

        return self.query

    def get_forecast_days(self, user_input_lower):
        week_days = self.language.week_days

        if "aujourd'hui" in user_input_lower:
            return 0
        elif "demain" in user_input_lower:
            return 1
        elif "après-demain" in user_input_lower or "après demain" in user_input_lower:
            return 2

        for i in range(2, 5):
            if f"dans {i} jour" in user_input_lower:
                return i

        today = datetime.now().weekday()
        if "week-end" in user_input_lower or "weekend" in user_input_lower:
            if "ce" in user_input_lower:
                days_to_saturday = (5 - today) % 7
                if days_to_saturday == 0 and datetime.now().hour >= 18:
                    self.is_weekend_query = True
                    return 1
                else:
                    self.is_weekend_query = True
                    return days_to_saturday

        for day_name, day_num in week_days.items():
            if day_name in user_input_lower:
                if day_num - today > 0:
                    return day_num - today
                elif today - day_num > 0:
                    return 7 - (today - day_num)
                else:
                    return 7

        if self.config.general.lang == "fr":
            if "semaine" in user_input_lower or "prochains jours" in user_input_lower:
                return -1
        else:
            if (
                "week" in user_input_lower
                or "next days" in user_input_lower
                or "following days" in user_input_lower
            ):
                return -1

        return 0

    def get_location_from_input(self, user_input):
        doc = self.nlp(user_input)

        for ent in doc.ents:
            if ent.label_ == "LOC":
                return ent.text

        location_prepositions = self.language.location_prepositions
        for i, token in enumerate(doc):
            if token.text.lower() in location_prepositions and i < len(doc) - 1:
                span_start = i + 1
                span_end = span_start + 1

                while span_end < len(doc) and (
                    doc[span_end].pos_ in ["PROPN", "NOUN", "ADJ"]
                    or doc[span_end].text
                    in ["de", "du", "des", "le", "la", "les", "l'"]
                ):
                    span_end += 1

                if span_end > span_start:
                    location_phrase = doc[span_start:span_end].text
                    if location_phrase:
                        return location_phrase

        return None

    def search(self, user_input=None, **kwargs):
        if not self.query:
            return {"success": False, "error": "Not a weather query"}

        results = {}

        if not self.location_name:
            default_loc = self.get_default_location()
            self.location_name = default_loc.get("city", "Unknown")

        if self.forecast_days == 0:
            self.weather_data = self.get_current_weather(self.location_name)
            if self.weather_data["success"]:
                results["weather"] = self.weather_data

        self.forecast_data = self.get_weather_forecast(self.location_name)
        if self.forecast_data["success"]:
            results["forecast"] = self.forecast_data

        return {"success": True, "data": results}

    def get_default_location(self):
        """Get location from IP address"""
        try:
            response = requests.get("https://ipapi.co/json/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    "city": data.get("city", "Unknown"),
                    "region": data.get("region", "Unknown"),
                    "country": data.get("country_name", "Unknown"),
                    "latitude": data.get("latitude"),
                    "longitude": data.get("longitude"),
                }
            return {"city": "Unknown", "region": "Unknown", "country": "Unknown"}
        except Exception as e:
            logger.error(f"Error getting location: {e}")
            return {"city": "Unknown", "region": "Unknown", "country": "Unknown"}

    def get_location_coords(self, location_name):
        try:
            url = f"https://nominatim.openstreetmap.org/search?q={location_name}&format=json&limit=1"
            headers = {
                "User-Agent": "LLMAssistant/1.0",
                "Accept-Language": f"{self.config.general.lang}-{self.config.general.lang.upper()},{self.config.general.lang};q=0.9"
                if self.config.general.lang == "fr"
                else "en-US,en;q=0.9",
            }

            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return {
                        "lat": float(data[0]["lat"]),
                        "lon": float(data[0]["lon"]),
                        "display_name": data[0]["display_name"],
                    }
            return None
        except Exception as e:
            logger.error(f"Error geocoding: {e}")
            return None

    def get_current_weather(self, location_name):
        try:
            coords = self.get_location_coords(location_name)
            if not coords:
                default_loc = self.get_default_location()
                coords = {
                    "lat": default_loc.get("latitude", 48.8566),
                    "lon": default_loc.get("longitude", 2.3522),
                }

            api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
            if not api_key:
                return {
                    "success": False,
                    "error": "No OpenWeatherMap API key configured",
                }

            lang_code = (
                self.config.general.lang
                if self.config.general.lang in ["fr", "en", "es", "de", "it"]
                else "en"
            )
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={coords['lat']}&lon={coords['lon']}&appid={api_key}&units=metric&lang={lang_code}"

            response = requests.get(url, timeout=5)

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Weather API error (code {response.status_code})",
                }

            data = response.json()

            return {
                "success": True,
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "min_temp": data["main"]["temp_min"],
                "max_temp": data["main"]["temp_max"],
                "pressure": data["main"]["pressure"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "icon": data["weather"][0]["icon"],
                "main": data["weather"][0]["main"],
                "wind_speed": data.get("wind", {}).get("speed", 0) * 3.6,
                "wind_direction": data.get("wind", {}).get("deg", 0),
                "clouds": data.get("clouds", {}).get("all", 0),
                "visibility": data.get("visibility", 0),
                "location": {
                    "name": data.get("name", location_name),
                    "country": data.get("sys", {}).get("country", ""),
                    "coordinates": coords,
                },
                "timestamp": int(time.time()),
            }

        except Exception as e:
            logger.error(f"Error getting current weather: {e}")
            return {"success": False, "error": f"Error: {str(e)}"}

    def get_weather_forecast(self, location_name, days=5):
        try:
            coords = self.get_location_coords(location_name)
            if not coords:
                default_loc = self.get_default_location()
                coords = {
                    "lat": default_loc.get("latitude", 48.8566),
                    "lon": default_loc.get("longitude", 2.3522),
                }

            api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
            if not api_key:
                return {
                    "success": False,
                    "error": "No OpenWeatherMap API key configured",
                }

            lang_code = (
                self.config.general.lang
                if self.config.general.lang in ["fr", "en", "es", "de", "it"]
                else "en"
            )
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={coords['lat']}&lon={coords['lon']}&appid={api_key}&units=metric&lang={lang_code}"

            response = requests.get(url, timeout=5)

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Forecast API error (code {response.status_code})",
                }

            data = response.json()
            forecast_by_day = {}
            city_name = data["city"]["name"]

            for item in data["list"]:
                dt = datetime.fromtimestamp(item["dt"])
                date_str = dt.strftime("%Y-%m-%d")

                if date_str not in forecast_by_day:
                    forecast_by_day[date_str] = {
                        "date": date_str,
                        "day_name": dt.strftime("%A"),
                        "min_temp": float("inf"),
                        "max_temp": float("-inf"),
                        "humidity_avg": 0,
                        "pressure_avg": 0,
                        "wind_speed_avg": 0,
                        "descriptions": [],
                        "icons": [],
                        "timestamps": [],
                        "forecast_points": 0,
                    }

                day_data = forecast_by_day[date_str]
                day_data["min_temp"] = min(
                    day_data["min_temp"], item["main"]["temp_min"]
                )
                day_data["max_temp"] = max(
                    day_data["max_temp"], item["main"]["temp_max"]
                )
                day_data["humidity_avg"] += item["main"]["humidity"]
                day_data["pressure_avg"] += item["main"]["pressure"]
                day_data["wind_speed_avg"] += (
                    item["wind"]["speed"] * 3.6
                )  # Convert m/s to km/h
                day_data["descriptions"].append(item["weather"][0]["description"])
                day_data["icons"].append(item["weather"][0]["icon"])
                day_data["timestamps"].append(item["dt"])
                day_data["forecast_points"] += 1

            daily_forecasts = []
            for date_str, day_data in sorted(forecast_by_day.items()):
                if day_data["forecast_points"] > 0:
                    day_data["humidity_avg"] /= day_data["forecast_points"]
                    day_data["pressure_avg"] /= day_data["forecast_points"]
                    day_data["wind_speed_avg"] /= day_data["forecast_points"]

                    desc_counter = Counter(day_data["descriptions"])
                    day_data["main_description"] = desc_counter.most_common(1)[0][0]
                    icon_counter = Counter(day_data["icons"])
                    day_data["main_icon"] = icon_counter.most_common(1)[0][0]

                    del day_data["descriptions"]
                    del day_data["icons"]
                    del day_data["forecast_points"]

                    daily_forecasts.append(day_data)

            return {
                "success": True,
                "location": {
                    "name": city_name,
                    "coordinates": coords,
                },
                "forecast": daily_forecasts[:days],
                "timestamp": int(time.time()),
                "source": "OpenWeatherMap",
            }

        except Exception as e:
            logger.error(f"Error getting weather forecast: {e}")
            return {"success": False, "error": f"Error: {str(e)}"}

    def format(self) -> str:
        if not self.weather_data and not self.forecast_data:
            return "No weather data available"

        weather_context = ""

        if self.weather_data and self.weather_data.get("success"):
            w = self.weather_data
            location_name = w["location"]["name"]

            if self.config.general.lang == "fr":
                weather_context += f"Météo actuelle à {location_name}:\n"
                weather_context += f"- Température: {w['temperature']:.1f}°C (ressentie {w['feels_like']:.1f}°C)\n"
                weather_context += f"- Conditions: {w['description']}\n"
                weather_context += f"- Humidité: {w['humidity']}%\n"
                weather_context += f"- Vent: {w['wind_speed']:.1f} km/h\n\n"
            else:
                weather_context += f"Current weather in {location_name}:\n"
                weather_context += f"- Temperature: {w['temperature']:.1f}°C (feels like {w['feels_like']:.1f}°C)\n"
                weather_context += f"- Conditions: {w['description']}\n"
                weather_context += f"- Humidity: {w['humidity']}%\n"
                weather_context += f"- Wind: {w['wind_speed']:.1f} km/h\n\n"

        if self.forecast_data and self.forecast_data.get("success"):
            f = self.forecast_data
            location_name = f["location"]["name"]

            if self.is_weekend_query and len(f["forecast"]) > self.forecast_days + 1:
                if self.config.general.lang == "fr":
                    weather_context += (
                        f"Prévisions météo pour {location_name} pour le week-end:\n"
                    )
                else:
                    weather_context += (
                        f"Weekend weather forecast for {location_name}:\n"
                    )

                if self.forecast_days < len(f["forecast"]):
                    day_saturday = f["forecast"][self.forecast_days]
                    date_obj_sat = datetime.strptime(day_saturday["date"], "%Y-%m-%d")
                    date_str_sat = date_obj_sat.strftime("%d/%m/%Y")

                    if self.config.general.lang == "fr":
                        weather_context += f"- Samedi {date_str_sat}: {day_saturday['min_temp']:.1f}°C à {day_saturday['max_temp']:.1f}°C, {day_saturday['main_description']}\n"
                    else:
                        weather_context += f"- Saturday {date_str_sat}: {day_saturday['min_temp']:.1f}°C to {day_saturday['max_temp']:.1f}°C, {day_saturday['main_description']}\n"

                if self.forecast_days + 1 < len(f["forecast"]):
                    day_sunday = f["forecast"][self.forecast_days + 1]
                    date_obj_sun = datetime.strptime(day_sunday["date"], "%Y-%m-%d")
                    date_str_sun = date_obj_sun.strftime("%d/%m/%Y")

                    if self.config.general.lang == "fr":
                        weather_context += f"- Dimanche {date_str_sun}: {day_sunday['min_temp']:.1f}°C à {day_sunday['max_temp']:.1f}°C, {day_sunday['main_description']}\n\n"
                    else:
                        weather_context += f"- Sunday {date_str_sun}: {day_sunday['min_temp']:.1f}°C to {day_sunday['max_temp']:.1f}°C, {day_sunday['main_description']}\n\n"

            elif self.forecast_days > 0 and self.forecast_days < len(f["forecast"]):
                day = f["forecast"][self.forecast_days]
                date_obj = datetime.strptime(day["date"], "%Y-%m-%d")
                date_str = date_obj.strftime("%d/%m/%Y")
                day_name = day["day_name"]

                if self.config.general.lang == "fr":
                    weather_context += f"Prévisions météo pour {location_name} ({day_name} {date_str}):\n"
                    weather_context += f"- Température: {day['min_temp']:.1f}°C à {day['max_temp']:.1f}°C\n"
                    weather_context += f"- Conditions: {day['main_description']}\n"
                    weather_context += (
                        f"- Humidité moyenne: {day['humidity_avg']:.0f}%\n"
                    )
                    weather_context += (
                        f"- Vent moyen: {day['wind_speed_avg']:.1f} km/h\n\n"
                    )
                else:
                    weather_context += f"Weather forecast for {location_name} ({day_name} {date_str}):\n"
                    weather_context += f"- Temperature: {day['min_temp']:.1f}°C to {day['max_temp']:.1f}°C\n"
                    weather_context += f"- Conditions: {day['main_description']}\n"
                    weather_context += (
                        f"- Average humidity: {day['humidity_avg']:.0f}%\n"
                    )
                    weather_context += (
                        f"- Average wind: {day['wind_speed_avg']:.1f} km/h\n\n"
                    )

            elif self.forecast_days == -1:
                if self.config.general.lang == "fr":
                    weather_context += f"Prévisions météo pour {location_name} pour les prochains jours:\n"
                else:
                    weather_context += (
                        f"Weather forecast for {location_name} for the next days:\n"
                    )

                for day in f["forecast"]:
                    date_obj = datetime.strptime(day["date"], "%Y-%m-%d")
                    date_str = date_obj.strftime("%d/%m/%Y")
                    day_name = day["day_name"]

                    if self.config.general.lang == "fr":
                        weather_context += f"- {day_name} {date_str}: {day['min_temp']:.1f}°C à {day['max_temp']:.1f}°C, {day['main_description']}\n"
                    else:
                        weather_context += f"- {day_name} {date_str}: {day['min_temp']:.1f}°C to {day['max_temp']:.1f}°C, {day['main_description']}\n"

                weather_context += "\n"

            else:
                if self.config.general.lang == "fr":
                    weather_context += f"Prévisions météo pour {location_name}:\n"
                else:
                    weather_context += f"Weather forecast for {location_name}:\n"

                days_to_show = min(2, len(f["forecast"]))
                for i in range(days_to_show):
                    day = f["forecast"][i]
                    date_obj = datetime.strptime(day["date"], "%Y-%m-%d")
                    date_str = date_obj.strftime("%d/%m/%Y")
                    day_name = day["day_name"]

                    if self.config.general.lang == "fr":
                        weather_context += f"- {day_name} {date_str}: {day['min_temp']:.1f}°C à {day['max_temp']:.1f}°C, {day['main_description']}\n"
                    else:
                        weather_context += f"- {day_name} {date_str}: {day['min_temp']:.1f}°C to {day['max_temp']:.1f}°C, {day['main_description']}\n"

                weather_context += "\n"

        return weather_context.strip()

    def search_terms(self, user_input):
        if self.location_name:
            return f"weather {self.location_name}"
        return "weather"
