import modal
import subprocess
import time

def pull_llava_model():
    server = subprocess.Popen(["ollama", "serve"])
    time.sleep(3)
    subprocess.run(["ollama", "pull", "llava"], check=True)
    server.terminate()

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("ollama")
    .run_function(pull_llava_model)
)

# Tên app này rất quan trọng, nó là "địa chỉ" để file khác gọi tới
app = modal.App("ollama-llava-server")

@app.cls(gpu="A10G", image=image, timeout=36000)
class OllamaServer:
    @modal.enter()
    def start_server(self):
        import ollama
        self.server = subprocess.Popen(["ollama", "serve"])
        time.sleep(3)
        self.client = ollama.Client()

    @modal.exit()
    def stop_server(self):
        self.server.terminate()

    @modal.method()
    def chat_vision(self, query: str, img_bytes):

        response = self.client.chat(
            model='llava',
            messages=[{'role': 'user', 'content': query, 'images': [img_bytes]}]
        )
        return response['message']['content']