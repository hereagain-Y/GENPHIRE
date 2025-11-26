#!/usr/bin/env python3
import os
import pandas as pd
import time
import argparse
from openai import OpenAI, OpenAIError

# ---------- Embedding Helper ----------
def get_gpt_embeddings(client, texts, max_retries=5):
    """Request embeddings from OpenAI API with retry on rate limits."""
    retries = 0
    while retries < max_retries:
        try:
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            return [d.embedding for d in response.data]
        except OpenAIError as e:
            if "rate_limit_exceeded" in str(e):
                wait_time = (2 ** retries)
                print(f"[Retry {retries+1}] Rate limit exceeded. Waiting {wait_time}s...")
                time.sleep(wait_time)
                retries += 1
            else:
                raise e
    print("⚠️ Max retries reached. Skipping batch.")
    return [None] * len(texts)

def generate_embeddings(input_file, column_name, output_file,batch_size=50):
    """Generate embeddings for a specified text column and save to output CSV."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    client = OpenAI(api_key=api_key)
    print(f"Loading {input_file} ...")
    df = pd.read_csv(input_file, usecols=['ID',column_name])
    df = df.dropna(subset=[column_name])
    print('df has total line of ', len(df))
    print(f"Generating embeddings for column: {column_name}")
    all_embeddings = []
    
    n = len(df)
    #print(df['trait_sequence_20'].iloc[0:10].isna().sum())
    for start in range(0, n, batch_size):
        batch_texts = df[column_name].iloc[start:start+batch_size].tolist()
        try:
            embeddings = get_gpt_embeddings(client, batch_texts)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"⚠️ Error on batch starting at index {start}: {e}")
            time.sleep(60)
            all_embeddings.extend([None]*len(batch_texts))

        print(f"Processed {min(start+batch_size, n)}/{n}")

    df['embedding_gpt'] = all_embeddings
    print(f"Saving to {output_file}")
    df.to_csv(output_file, index=False)
    print(" Embedding generation complete.")

# ---------- CLI Interface ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GPT embeddings for any CSV text column.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--column", required=True, help="Column name containing text.")
    parser.add_argument("--output", required=True, help="Path to save output CSV.")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for embedding requests.")

    args = parser.parse_args()
    generate_embeddings(args.input, args.column, args.output,  args.batch_size)
