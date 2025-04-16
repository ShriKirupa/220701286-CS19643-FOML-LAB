import subprocess
import re

def call_deepseek(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "deepseek-r1:latest"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True  # Ensures an exception is raised for non-zero exit code
        )

        response = result.stdout.strip()

        # Print full response for debugging
        print("\n🔍 Full DeepSeek Response:\n", response)

        # Optional: extract code block if it exists (matching any language)
        code_match = re.search(r"```(.*?)```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            return {"full": response, "code": code}
        else:
            return {"full": response, "code": None}
    
    except subprocess.CalledProcessError as e:
        print(f"Error calling DeepSeek: {e}")
        return {"full": "⚠️ Error occurred while retrieving the response.", "code": None}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"full": "⚠️ An unexpected error occurred.", "code": None}
