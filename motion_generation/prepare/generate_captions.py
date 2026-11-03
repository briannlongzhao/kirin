"""Caption every clip's motion with Gemini.

For each clip folder under --data_dir, the cropped CLIP_ID_rgb.mp4 is captioned
with six prompts, each with a random style example from --example_captions.
Captions are tokenized with spaCy and written to texts_gemini.txt as
"caption#word/POS ...#0.0#0.0" lines. Requires GEMINI_API_KEY.
"""

import argparse
import os
from pathlib import Path
from random import choice
from time import sleep

import spacy
from google import genai
from google.genai import types
from tqdm import tqdm

from data_loaders.humanml.data.dataset import get_all_sequence_dirs

SYSTEM_PROMPT = ("You are annotating a text-to-motion generation dataset. You will be given a video clip of animal "
                 "motion, and you will be asked to write concise motion captions suitable for training a "
                 "text-to-motion generation model. ")
VIDEO_CAPTION_PROMPTS = [
    "Describe the {}'s motion in a single short sentence.",
    "The {} is doing what kind of motion in the video? Answer in one short sentence.",
    "Use simple sentences to describe the {}'s motion and global movement in the video.",
    "What is the main action of the {} in the video? Be specific about **speed, style, and sequence of actions**.",
    "Describe the {}'s motion in this video. Focus on the detailed motion.",
    "Write a short caption of the {} motion in this video that describes the **motion type** and **global movement path**.",
]


def process_caption(caption, nlp):
    """Format a caption as a HumanML3D text line with lemmatized nouns and verbs."""
    caption = caption.lower().strip()
    data = caption + "#"
    for token in nlp(caption):
        if token.pos_ in ["NOUN", "VERB"] and token.text != "left":
            data += f"{token.lemma_}/{token.pos_} "
        elif not token.text.isalpha():
            continue
        else:
            data += f"{token.text}/{token.pos_} "
    return data.strip() + "#0.0#0.0"


def caption_video(client, model, video_file, prompt):
    config = types.GenerateContentConfig(
        max_output_tokens=1024,
        temperature=0.85,
        media_resolution="MEDIA_RESOLUTION_LOW",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        system_instruction=[types.Part.from_text(text=SYSTEM_PROMPT)],
    )
    contents = [video_file, types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    text = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
        if chunk.text:
            text += chunk.text
    return text.replace("\n", " ").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Dataset root organized as CATEGORY/VIDEO_ID/CLIP_ID.")
    parser.add_argument("--example_captions", required=True,
                        help="Text file with one example caption per line, used to set the caption style.")
    parser.add_argument("--model", default="gemini-flash-latest")
    args = parser.parse_args()

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    nlp = spacy.load("en_core_web_sm")
    examples = [line.strip() for line in open(args.example_captions) if line.strip()]
    for sequence_dir in tqdm(get_all_sequence_dirs(args.data_dir)):
        out_path = os.path.join(sequence_dir, "texts_gemini.txt")
        if os.path.exists(out_path):
            continue
        category = Path(sequence_dir).parent.parent.name
        clip_id = os.path.basename(sequence_dir)
        video_file = client.files.upload(file=os.path.join(sequence_dir, f"{clip_id}_rgb.mp4"))
        lines = []
        for prompt in VIDEO_CAPTION_PROMPTS:
            prompt = prompt.format(category) + " Following the style of this example: " + choice(examples)
            while True:
                try:
                    response = caption_video(client, args.model, video_file, prompt)
                    break
                except Exception as e:
                    print(e)
                    sleep(2)
            lines.append(process_caption(response, nlp))
        with open(out_path, "w") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    main()
