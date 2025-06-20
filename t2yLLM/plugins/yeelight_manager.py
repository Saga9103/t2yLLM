import yaml
from pathlib import Path
from yeelight import discover_bulbs, Bulb
import argparse
from typing import Dict
import time
from datetime import datetime


class YeelightManager:
    def __init__(self, config_path: str = "yeelightConfig.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> Dict:
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {"rooms": {}, "bulbs": {}, "settings": {}}
        return {
            "rooms": {},
            "bulbs": {},
            "settings": {
                "default_transition": "smooth",
                "default_duration": 300,
                "auto_discover_on_start": True,
                "reconnect_attempts": 2,
                "command_timeout": 5,
            },
            "last_update": datetime.now().isoformat(),
        }

    def save_config(self):
        self.config["last_update"] = datetime.now().isoformat()
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved as {self.config_path}")

    def discover(self):
        print("Discovering Yeelight devices")
        discovered = discover_bulbs()

        if not discovered:
            print(
                "No Yeelight devices found. Check network connection or allow them from the yeelight app"
            )
            return

        print(f"\nFound {len(discovered)} device(s):")
        new_bulbs = 0

        for i, bulb_info in enumerate(discovered):
            cap = bulb_info["capabilities"]
            bulb_id = cap["id"]
            print(f"\n{i + 1}. Bulb ID: {bulb_id}")
            print(f"   IP: {bulb_info['ip']}")
            print(f"   Model: {cap['model']}")
            print(f"   Power: {cap['power']}")
            print(f"   Brightness: {cap['bright']}%")

            if bulb_id not in self.config["bulbs"]:
                self.config["bulbs"][bulb_id] = {
                    "ip": bulb_info["ip"],
                    "model": cap["model"],
                    "name": f"Bulb_{bulb_id[-4:]}",
                    "id": bulb_id,
                }
                print("   -> Added to configuration")
                new_bulbs += 1
            else:
                if self.config["bulbs"][bulb_id]["ip"] != bulb_info["ip"]:
                    self.config["bulbs"][bulb_id]["ip"] = bulb_info["ip"]
                    print("   -> Updated IP address")

        if new_bulbs > 0:
            self.save_config()
            print(f"\nAdded {new_bulbs} new devices.")
        else:
            print("\nNo new devices found.")

    def list_bulbs(self):
        if not self.config["bulbs"]:
            print("No devices configured. Run 'discover' first.")
            return

        print("\nConfigured Yeelight device(s) :")
        print("-" * 60)
        for bulb_id, info in self.config["bulbs"].items():
            print(f"\nName: {info['name']}")
            print(f"ID: {bulb_id}")
            print(f"IP: {info['ip']}")
            print(f"Model: {info['model']}")
            rooms = []
            for room, bulb_ids in self.config["rooms"].items():
                if bulb_id in bulb_ids:
                    rooms.append(room)
            if rooms:
                print(f"Rooms: {', '.join(rooms)}")
            else:
                print("Rooms: Not assigned")

    def list_rooms(self):
        if not self.config["rooms"]:
            print("No rooms configured.")
            return
        print("\nConfigured rooms:")
        print("-" * 60)
        for room, bulb_ids in self.config["rooms"].items():
            print(f"\n{room}:")
            if bulb_ids:
                for bulb_id in bulb_ids:
                    if bulb_id in self.config["bulbs"]:
                        bulb_info = self.config["bulbs"][bulb_id]
                        print(f"  - {bulb_info['name']} (IP: {bulb_info['ip']})")
            else:
                print("  (empty)")

    def add_room(self, room_name: str):
        if room_name in self.config["rooms"]:
            print(f"Room '{room_name}' already exists.")
            return

        self.config["rooms"][room_name] = []
        self.save_config()
        print(f"Room '{room_name}' added.")

    def remove_room(self, room_name: str):
        if room_name not in self.config["rooms"]:
            print(f"Room '{room_name}' not found.")
            return

        del self.config["rooms"][room_name]
        self.save_config()
        print(f"Room '{room_name}' removed.")

    def assign_bulb_to_room(self, bulb_id: str, room_name: str):
        actual_bulb_id = None
        for bid, info in self.config["bulbs"].items():
            if bid == bulb_id or info["name"].lower() == bulb_id.lower():
                actual_bulb_id = bid
                break

        if not actual_bulb_id:
            print(
                f"Device '{bulb_id}' not found. Run 'discover' first or check the device name/ID."
            )
            return

        if room_name not in self.config["rooms"]:
            print(f"Room '{room_name}' not found. Create it first with 'add-room'.")
            return

        if actual_bulb_id not in self.config["rooms"][room_name]:
            self.config["rooms"][room_name].append(actual_bulb_id)
            self.save_config()
            bulb_name = self.config["bulbs"][actual_bulb_id]["name"]
            print(f"Yeelight '{bulb_name}' assigned to room '{room_name}'.")
        else:
            print(f"Yeelight device already in a room : '{room_name}'.")

    def remove_bulb_from_room(self, bulb_id: str, room_name: str):
        actual_bulb_id = None
        for bid, info in self.config["bulbs"].items():
            if bid == bulb_id or info["name"].lower() == bulb_id.lower():
                actual_bulb_id = bid
                break

        if not actual_bulb_id:
            print(f"Yeelight '{bulb_id}' not found.")
            return

        if room_name not in self.config["rooms"]:
            print(f"Room '{room_name}' not found.")
            return

        if actual_bulb_id in self.config["rooms"][room_name]:
            self.config["rooms"][room_name].remove(actual_bulb_id)
            self.save_config()
            bulb_name = self.config["bulbs"][actual_bulb_id]["name"]
            print(f"Device '{bulb_name}' removed from room '{room_name}'.")
        else:
            print(f"Device not found in room '{room_name}'.")

    def rename_bulb(self, bulb_id: str, new_name: str):
        actual_bulb_id = None
        for bid, info in self.config["bulbs"].items():
            if bid == bulb_id or info["name"].lower() == bulb_id.lower():
                actual_bulb_id = bid
                break

        if not actual_bulb_id:
            print(f"Device '{bulb_id}' not found.")
            return

        old_name = self.config["bulbs"][actual_bulb_id]["name"]
        self.config["bulbs"][actual_bulb_id]["name"] = new_name
        self.save_config()
        print(f"Device '{old_name}' renamed to '{new_name}'.")

    def test_bulb(self, bulb_id: str):
        actual_bulb_id = None
        bulb_info = None
        for bid, info in self.config["bulbs"].items():
            if bid == bulb_id or info["name"].lower() == bulb_id.lower():
                actual_bulb_id = bid
                bulb_info = info
                break

        if not actual_bulb_id:
            print(f"Device '{bulb_id}' not found.")
            return

        print(f"Testing {bulb_info['name']}...")

        try:
            bulb = Bulb(bulb_info["ip"])
            props = bulb.get_properties()
            original_power = props.get("power", "off")

            for i in range(3):
                bulb.turn_off()
                time.sleep(1.0)
                bulb.turn_on()
                time.sleep(1.0)

            if original_power == "off":
                bulb.turn_off()

            print("Test completed!")
        except Exception as e:
            print(f"Error testing device : {e}")
            print("Make sure the device is powered on and connected to the network.")

    def test_room(self, room_name: str):
        if room_name not in self.config["rooms"]:
            print(f"Room '{room_name}' not found.")
            return

        bulb_ids = self.config["rooms"][room_name]
        if not bulb_ids:
            print(f"No Yeelight device in cuurent room '{room_name}'.")
            return

        print(f"Testing all devices in room '{room_name}'...")

        for bulb_id in bulb_ids:
            if bulb_id in self.config["bulbs"]:
                self.test_bulb(bulb_id)
                time.sleep(1)

    def interactive_setup(self):
        print("\n" + "=" * 60)
        print("YEELIGHT SETUP WIZARD")
        print("=" * 60 + "\n")
        self.discover()

        if not self.config["bulbs"]:
            return

        print("\n" + "-" * 40)
        print("Step 1: Rename Yeelight devices")
        print("-" * 40)
        print("\nWould you like to rename your device ? (y/n) : ", end="")
        if input().lower() == "y":
            for bulb_id, info in list(self.config["bulbs"].items()):
                print(f"\nCurrent name: {info['name']}")
                print(f"IP: {info['ip']}")
                print("Enter new name (or press Enter to ignore) : ", end="")
                new_name = input().strip()
                if new_name:
                    self.rename_bulb(bulb_id, new_name)

        print("\n" + "-" * 40)
        print("Step 2: Create Rooms")
        print("-" * 40)
        print("\nWould you like to create rooms? (y/n): ", end="")
        if input().lower() == "y":
            while True:
                print("\nEnter room name (or press Enter to finish): ", end="")
                room_name = input().strip()
                if not room_name:
                    break
                self.add_room(room_name)

                print(f"\nAssign Yeelight devices to '{room_name}'?")
                for bulb_id, info in self.config["bulbs"].items():
                    print(f"Add '{info['name']}'? (y/n): ", end="")
                    if input().lower() == "y":
                        self.assign_bulb_to_room(bulb_id, room_name)

        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print("\nYour configuration has been saved.")


def main():
    parser = argparse.ArgumentParser(description="Yeelight Management Tool")
    parser.add_argument(
        "-c",
        "--config",
        default="yeelightConfig.yaml",
        help="Path to configuration file",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    subparsers.add_parser("discover", help="Discover Yeelight bulbs on the network")
    subparsers.add_parser("list-bulbs", help="List all configured bulbs")
    subparsers.add_parser("list-rooms", help="List all rooms and their bulbs")

    room_add = subparsers.add_parser("add-room", help="Add a new room")
    room_add.add_argument("name", help="Room name")
    room_remove = subparsers.add_parser("remove-room", help="Remove a room")
    room_remove.add_argument("name", help="Room name")

    assign = subparsers.add_parser("assign", help="Assign a bulb to a room")
    assign.add_argument("bulb", help="Bulb ID or name")
    assign.add_argument("room", help="Room name")

    unassign = subparsers.add_parser("unassign", help="Remove a bulb from a room")
    unassign.add_argument("bulb", help="Bulb ID or name")
    unassign.add_argument("room", help="Room name")

    rename = subparsers.add_parser("rename", help="Rename a bulb")
    rename.add_argument("bulb", help="Bulb ID or current name")
    rename.add_argument("new_name", help="New name for the bulb")

    test_bulb = subparsers.add_parser("test-bulb", help="Test a bulb by blinking it")
    test_bulb.add_argument("bulb", help="Bulb ID or name")

    test_room = subparsers.add_parser("test-room", help="Test all bulbs in a room")
    test_room.add_argument("room", help="Room name")

    subparsers.add_parser("setup", help="Run interactive setup wizard")

    args = parser.parse_args()

    manager = YeelightManager(args.config)

    if args.command == "discover":
        manager.discover()
    elif args.command == "list-bulbs":
        manager.list_bulbs()
    elif args.command == "list-rooms":
        manager.list_rooms()
    elif args.command == "add-room":
        manager.add_room(args.name)
    elif args.command == "remove-room":
        manager.remove_room(args.name)
    elif args.command == "assign":
        manager.assign_bulb_to_room(args.bulb, args.room)
    elif args.command == "unassign":
        manager.remove_bulb_from_room(args.bulb, args.room)
    elif args.command == "rename":
        manager.rename_bulb(args.bulb, args.new_name)
    elif args.command == "test-bulb":
        manager.test_bulb(args.bulb)
    elif args.command == "test-room":
        manager.test_room(args.room)
    elif args.command == "setup":
        manager.interactive_setup()
    else:
        parser.print_help()


if __name__ != "__main__":
    raise ImportError("You are trying to import a configuration script")

if __name__ == "__main__":
    main()
