from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
text = '''
# 1. Define the Parent Class
class Vehicle:
    """A blueprint representing a general vehicle."""
    
    def __init__(self, brand: str, model: str, year: int):
        # Instance attributes
        self.brand = brand
        self.model = model
        self.year = year
        self.odometer = 0  # Default value for a new attribute

    def display_info(self):
        """Displays formatted information about the vehicle."""
        print(f"{self.year} {self.brand} {self.model}")

    def drive(self, miles: float):
        """Simulates driving and adds to the odometer."""
        if miles > 0:
            self.odometer += miles
            print(f"Drove {miles} miles. Total odometer: {self.odometer} miles.")
        else:
            print("You can't drive backwards on the odometer!")


# 2. Define a Child Class (Inheritance)
class ElectricCar(Vehicle):
    """A specific type of vehicle that inherits from Vehicle."""
    
    def __init__(self, brand: str, model: str, year: int, battery_size: int):
        # Initialize attributes from the parent class
        super().__init__(brand, model, year)
        # Unique attribute for this specific child class
        self.battery_size = battery_size

    def charge_battery(self):
        """Simulates charging the electric car."""
        print(f"Charging the {self.model}'s {self.battery_size} kWh battery to 100%...")


# ==========================================
# 3. Use the Classes (Creating Objects)
# ==========================================
if __name__ == "__main__":
    print("--- Testing Parent Class (Vehicle) ---")
    # Instantiate a generic vehicle object
    my_truck = Vehicle("Ford", "F-150", 2022)
    my_truck.display_info()
    my_truck.drive(150.5)

    print("\n--- Testing Child Class (ElectricCar) ---")
    # Instantiate an electric car object
    my_ev = ElectricCar("Tesla", "Model 3", 2024, 75)
    my_ev.display_info()      # Uses inherited method from Vehicle
    my_ev.drive(45)           # Uses inherited method from Vehicle
    my_ev.charge_battery()    # Uses the unique method only found in ElectricCar
'''


splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size = 350,
    language = Language.PYTHON,   # since we have a pytho code here
    chunk_overlap = 0
)

result = splitter.split_text(text)