from huggingface_hub import login
login(token='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')

from transformers import MarianMTModel, MarianTokenizer
import torch
import re
import sys
import os
import time

class NorwegianTranslator:
   def __init__(self, device_type="auto"):
       print("Initializing translator...")
       self.model_name = "Helsinki-NLP/opus-mt-gmq-en"
       self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
       self.model = MarianMTModel.from_pretrained(self.model_name)
       
       if device_type == "auto":
           self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
       else:
           self.device = torch.device(device_type)
           
       print(f"Using device: {self.device}")
       self.model.to(self.device)
       print("Initialization complete")

   def translate(self, text):
       start_time = time.time()
       inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
       translated = self.model.generate(**inputs)
       result = self.tokenizer.decode(translated[0], skip_special_tokens=True)
       end_time = time.time()
       return result, end_time - start_time

   def process_transcript(self, input_file):
       base_name = os.path.splitext(input_file)[0]
       output_file = f"{base_name}_translated.txt"
       total_time = 0
       
       with open(input_file, 'r', encoding='utf-8') as f:
           text = f.read()
           sentences = [s.strip() for s in re.split('[.!?]', text) if s.strip()]
           print(f"Found {len(sentences)} sentences")
           
       with open(output_file, 'w', encoding='utf-8') as f:
           for i, sentence in enumerate(sentences, 1):
               translation, time_taken = self.translate(sentence)
               total_time += time_taken
               print(f"Sentence {i}/{len(sentences)} - Time: {time_taken:.2f}s", end='\r')
               f.write(f"[Original {i}]: {sentence}\n")
               f.write(f"[English {i}]: {translation}\n\n")
               
       print(f"\nTranslation complete!")
       print(f"Total time: {total_time:.2f}s")
       print(f"Average time per sentence: {total_time/len(sentences):.2f}s")

def main():
   if len(sys.argv) < 2:
       print("Usage: python translator.py <input_file> [cpu|mps]")
       sys.exit(1)
       
   input_file = sys.argv[1]
   device = sys.argv[2] if len(sys.argv) > 2 else "auto"
   
   if not os.path.exists(input_file):
       print(f"Error: File '{input_file}' not found")
       sys.exit(1)
       
   translator = NorwegianTranslator(device)
   translator.process_transcript(input_file)

if __name__ == "__main__":
   main()
