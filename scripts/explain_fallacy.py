import shap
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import matplotlib.pyplot as plt

# Model setup
model_name = "RowdyI7er/DebateLLM"
subfolder = "debate_fallacy_model"

print(f"Loading model and tokenizer for SHAP analysis...")
tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder=subfolder)
model = AutoModelForSequenceClassification.from_pretrained(model_name, subfolder=subfolder)

# Define pipeline for shap
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, top_k=None)

# Initialize SHAP explainer
print("Initializing SHAP explainer (this could take a moment)...")
explainer = shap.Explainer(pipe)

def explain(text):
    print(f"\nAnalyzing: \"{text}\"")
    
    # Calculate SHAP values
    shap_values = explainer([text])
    
    # Generate label-specific reports
    prediction = pipe(text)[0]
    top_label = prediction[0]['label']
    top_score = prediction[0]['score']
    
    print(f"Prediction: {top_label} ({top_score:.2%})")
    
    # Create HTML visualization for the top predicted label
    # SHAP for text classification requires slicing by label index.
    # Let's find the index of the top predicted label.
    label_idx = 0
    for idx, label in model.config.id2label.items():
        if label == top_label:
            label_idx = idx
            break

    # Save visualization to HTML
    output_dir = os.path.join("reports", "explanations")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    safe_name = text[:20].replace(" ", "_").replace("/", "_").replace("\"", "")
    output_path = os.path.join(output_dir, f"explanation_{safe_name}.html")
    
    print(f"Generating heatmap for label '{top_label}'...")
    # shap.plots.text allows saving to HTML directly by using IPython interface if possible, 
    # but for script usage, we'll manually dump the plot structure.
    
    # To save as HTML, we'll use shap_values[sentence_idx, label_idx]
    # Note: text_plot doesn't have a direct 'save' but it renders perfectly in notebooks.
    # In CLI, we'll use a hack to save it.
    try:
        plot_html = shap.plots.text(shap_values[0, :, label_idx], display=False)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(plot_html)
        print(f"✅ Explanation saved: {output_path}")
    except Exception as e:
        print(f"❌ Error saving HTML: {e}")
        # Fallback to bar plot in matplotlib
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values[0, :, label_idx], show=False)
        plt.tight_layout()
        plt.savefig(output_path.replace(".html", ".png"))
        print(f"✅ Fallback PNG saved: {output_path.replace('.html', '.png')}")

if __name__ == "__main__":
    # Sample Test Sentence
    test_text = "You can't trust Professor Miller's research on environmental science; he's a known liar and probably hates his own country."
    
    # Run explanation
    explain(test_text)
    
    import sys
    if len(sys.argv) > 1:
        custom_text = " ".join(sys.argv[1:])
        explain(custom_text)
