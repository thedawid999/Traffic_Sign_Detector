#----- Converts labels from GTSDB-Format to YOLO-Format and creates seperate files for each image -----#

IMAGE_WIDTH = 1360
IMAGE_HEIGHT = 800
INPUT_FILE = "./data/labels/yolo/gt.txt"
OUTPUT_FOLDER = "./data/labels/yolo/"

def gtsdb():
    # a Set that contains all lines with the same IDs to store them in one file
    # if there's only 1 line (no multiple with same IDs) it will be also saved, as one only
    same_id = set()
    # a List that contains all of same_id sets
    grouped = []
    with open(INPUT_FILE) as gt:
        lines = [line.rstrip('\n') for line in gt]

        # Save the first line
        same_id.add(lines[0])

    for i in range(1, len(lines)):
        current = lines[i]
        previous = lines[i-1]

        # Check if the IDs are the same
        if current[:5] == previous[:5]:
            same_id.add(current)
        else:
            # The ID has changed, so save the current group
            grouped.append(list(same_id))
            # And start a new group with the current line
            same_id = {current}

    # After the loop, save the very last group
    if same_id:
        grouped.append(list(same_id))
                
    # Go through every entry of group
    for entry in grouped:
        filename = entry[0][:5]
            
        # Create a file for each entry
        with open(OUTPUT_FOLDER+"db"+filename+".txt", "w") as f:

            # Go through every line in entry (all have same IDs)
            for line in entry:

                # Split each line
                parts = line.split(";")

                # Calculate the YOLO-Format
                x_center = round((int(parts[1])+int(parts[3]))/2/IMAGE_WIDTH, 5)
                y_center = round((int(parts[2])+int(parts[4]))/2/IMAGE_HEIGHT, 5)
                width = round((int(parts[3])-int(parts[1]))/IMAGE_WIDTH, 5)
                height = round((int(parts[4])-int(parts[2]))/IMAGE_HEIGHT, 5)
                #class_id = "0"
                
                class_id = parts[5]

                # Write to the new file
                f.write(class_id + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height)+"\n")

def gtsrb():
    with open(INPUT_FILE, "r") as f:
        lines = f.readlines()

# optional: skip the header if included
    if lines[0].lower().startswith("filename"):
        lines = lines[1:]

    for line in lines:
        parts = line.split()
        filename = parts[0].replace(".ppm", ".txt")
        width = float(parts[1])
        height = float(parts[2])
        x1 = float(parts[3])
        y1 = float(parts[4])
        x2 = float(parts[5])
        y2 = float(parts[6])
        class_id = parts[7]

        # convert to yolo format
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height

        yolo_line = f"{class_id} {x_center:.5f} {y_center:.5f} {w:.5f} {h:.5f}\n"

        with open(OUTPUT_FOLDER+filename+".txt", "w") as out_f:
            out_f.write(yolo_line)

if __name__ == "__main__":
    gtsdb()