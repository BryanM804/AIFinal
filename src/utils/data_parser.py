
DIGIT_PATH = "./data/digitdata"
FACE_PATH = "./data/facedata"

def parse_numbers(file_name):
    # 28 x 28 images
    trans_table = str.maketrans({" ": "0.0", "\n": "0.0", "+": "1.0", "#": "1.0"})
    numbers = []
    with open(f"{DIGIT_PATH}/{file_name}", "r") as file:
        lines = file.read().splitlines()
        count = 0
        number = []
        for line in lines:
            count += 1
            chars = list(line)
            for char in chars:
                f = char.translate(trans_table)
                number.append(f)

            if count == 28:
                count = 0
                numbers.append(number)
                number = []

    return numbers

def parse_faces(file_name):
    # 60 x 74 arrays
    pass

def attach_labels(arr, label_file_name, digits):
    path = f"{DIGIT_PATH}/{label_file_name}" if digits else f"{FACE_PATH}/{label_file_name}"

    new_numbers = []

    with open(path, "r") as file:
        for x, line in enumerate(file.read().splitlines()):
            new_numbers.append((arr[x], int(line)))

    return new_numbers

def get_labels(label_file_name, digits):
    path = f"{DIGIT_PATH}/{label_file_name}" if digits else f"{FACE_PATH}/{label_file_name}"
    
    with open(path, "r") as file:
        labels = file.read().splitlines()

    return labels

if __name__ == "__main__":
    # testing
    parsed_numbers = parse_numbers("trainingimages")
    labeled_numbers = attach_labels(parsed_numbers, "traininglabels", True)
    for i in range(28 * 28): # couldnt bother opening a calc (short for calculator)
        print(labeled_numbers[0][0][i], end="")
        if i % 28 == 0: print("\n")

    # print(labeled_numbers[0][0])
    print(labeled_numbers[0][1])