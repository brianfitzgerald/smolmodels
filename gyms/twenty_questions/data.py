from dataclasses import dataclass
from typing import List, Tuple
import nltk

@dataclass
class WordVariants:
    words: List[str]
    pos_tags: List[List[Tuple[str, str]]]

    @classmethod
    def from_list(cls, words_list: List[str]):
        pos_tags = [
            nltk.pos_tag(nltk.word_tokenize(word.lower())) for word in words_list
        ]
        return cls(words=words_list, pos_tags=pos_tags)

    @classmethod
    def from_str(cls, words_str: str):
        words_list = words_str.split(";")
        return cls.from_list(words_list)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        assert 0 <= idx < len(self.words), f"Index {idx} out of range"
        return self.words[idx]

    def json(self):
        return self.words.copy()

    def __str__(self):
        return f"({', '.join(self.words)})"

    def __repr__(self) -> str:
        return f"WordVariants([{', '.join(self.words)}])"


DEFAULT_OBJECT_DICT = {
    "Sports": [
        "Basketball",
        "Football",
        "Baseball",
        "Soccer ball",
        "Golf ball",
        "Tennis ball",
        "Volleyball",
        "Tennis racket",
        "Baseball bat",
        "Helmet",
    ],
    "Animals": [
        "Cat",
        "Dog",
        "Horse",
        "Cow",
        "Sheep",
        "Rabbit",
        "Lion",
        "Tiger",
        "Bear",
        "Elephant",
    ],
    "Fruits": [
        "Apple",
        "Banana",
        "Orange",
        "Strawberry",
        "Grape",
        "Watermelon",
        "Pineapple",
        "Mango",
        "Cantaloupe",
        "Peach",
    ],
    "Vehicles": [
        "Car",
        "Truck",
        "Motorcycle",
        "Boat",
        "Airplane;Plane",
        "Train",
        "Bus",
        "Helicopter",
        "Scooter",
        "Ship",
    ],
    "Clothes": [
        "Shirt",
        "Pants;Pant;Pair of pants",
        "Jacket",
        "Dress",
        "Skirt",
        "Belt",
        "Shoes;Shoe;Pair of shoes",
        "Boots;Boot;Pair of boots",
        "Socks;Sock;Pair of socks",
        "Hat",
        "Scarf",
    ],
    "Electronics": [
        "Computer",
        "Smartphone",
        "Television;TV",
        "Headphone;Headphones;Pair of headphones",
        "Monitor;Computer monitor",
        "Camera",
        "Microwave;Microwave oven",
        "Refrigerator",
        "Blender",
        "Computer keyboard;Keyboard",
    ],
    "Musical Instruments": [
        "Piano",
        "Guitar",
        "Drum;Drums",
        "Violin",
        "Saxophone",
        "Flute",
        "Trumpet",
        "Clarinet",
        "Harp",
        "Trombone",
    ],
    "Furniture": [
        "Chair",
        "Table",
        "Bed",
        "Desk",
        "Couch",
        "Dresser",
        "Bookcase",
        "Nightstand",
        "Mattress",
        "Pillow",
    ],
    "Office Supplies": [
        "Pen",
        "Paper;Piece of paper",
        "Stapler",
        "Printer",
        "Calculator",
        "Battery;Battery pack;Pack of batteries",
        "Toothbrush",
        "Toothpaste",
        "Pencil",
        "Sharpie",
        "Scissors;Pair of scissors",
        "Key",
        "Diary",
        "Calendar",
    ],
    "Vegetables": [
        "Carrot",
        "Potato",
        "Broccoli",
        "Tomato",
        "Onion",
        "Spinach",
        "Corn",
        "Peas;Pea",
        "Celery",
        "Cucumber",
    ],
    "Art": [
        "Painting;Canvas painting;Oil painting;Watercolor painting",
        "Paintbrush",
        "Canvas;Painting canvas",
        "Eraser;Pencil eraser",
        "Marker",
        "Glue;Glue stick;Bottle of glue",
        "Sculpture",
    ],
    "Kitchen Tools": [
        "Knife",
        "Spoon",
        "Fork",
        "Plate",
        "Bowl",
        "Cooking pot;Pot",
        "Pan;Saucepan;Frying pan",
        "Cup",
        "Chopstick;Chopsticks;Pair of chopsticks",
        "Whisk",
    ],
    "Nature": [
        "Rock",
        "Tree",
        "Bush",
        "Mountain",
        "Forest",
        "Ocean",
        "Sea",
        "Lake",
        "River",
        "Meteorite",
        "Cactus",
    ],
    "Toys": [
        "Lego;Lego set",
        "Doll;Toy doll;Plush doll",
        "Kite",
        "Puzzle;Jigsaw puzzle",
        "Stuffed animal",
    ],
    "Jewelry": [
        "Earring;Earrings;Pair of earrings",
        "Necklace",
        "Bracelet",
        "Ring",
        "Brooch",
        "Hairclip",
        "Pendant",
        "Watch",
        "Locket",
    ],
    "Garden Supplies": [
        "Gloves;Glove;Pair of gloves",
        "Shovel",
        "Rake",
        "Watering can",
        "Lawn mower",
    ],
    "Tools": [
        "Hammer",
        "Screwdriver",
        "Wrench",
        "Saw",
        "Pliers;plier;Pair of pliers",
        "Drill",
    ],
}

INVALID_QUESTION = "Is this a valid question?\n"
INITIAL_STR = "Questions:\n"


def get_default_word_list() -> List[WordVariants]:
    word_list = []
    for _, words in DEFAULT_OBJECT_DICT.items():
        word_list.extend(map(lambda x: WordVariants.from_str(x), words))
    return word_list
