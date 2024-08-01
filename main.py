import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import streamlit as st
from PIL import ImageDraw, ImageFont
import random

# Set up the model and processor
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_id).to(device)


def load_image(image_path):
    return Image.open(image_path)


def process_image(image, text):
    inputs = processor(images=image, text=[
                       text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        [inputs.input_ids[0]] * len([text]),
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )[0]
    return results


def generate_color(label):
    random.seed(hash(label))
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def draw_boxes(image, results):
    draw = ImageDraw.Draw(image)
    object_count = {}

    image_width, image_height = image.size
    thickness = max(1, image_width // 100)
    font_size = max(10, image_width // 50)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # Convert tensor to list and round
        box = [round(i.item(), 2) for i in box]
        color = generate_color(label)
        draw.rectangle(box, outline=color, width=thickness)
        draw.text((box[0], box[1] - font_size),
                  f"{label} {round(score.item(), 3)}", fill=color, font=font)

        if label not in object_count:
            object_count[label] = 0
        object_count[label] += 1

    return image, object_count

def main():
    st.title("Zero-Shot Object Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        image = load_image(uploaded_file)

        text_query = st.text_input(
            "Enter the object description (e.g., 'a cat.', 'a remote control.', 'chocolate pieces.')", "chocolate pieces.")

        if st.button("Detect Objects"):
            st.write("Detecting objects...")

            results = process_image(image, text_query)
            image_with_boxes, object_count = draw_boxes(image, results)

            st.image(image_with_boxes, caption='Processed Image', use_column_width=True)

            st.write("Detected objects:")
            for label, count in object_count.items():
                st.write(f"{label}: {count}")

if __name__ == "__main__":
    main()
