import json
# pyrefly: ignore [missing-import]
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import models

# pyrefly: ignore [missing-import]
from ollama import chat


class Agent():
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.documents = []
        text_document_path = "result_document.json"
        with open(text_document_path, "r+", encoding="utf-8") as document_file:
            old_data = json.load(document_file)

        for i,video_descrip in enumerate(old_data):
            for clip_index in range(len(video_descrip["encoder"])):
                clip_descrip = video_descrip["encoder"][str(clip_index)]

                audio_text = clip_descrip["audio_text"].replace("\n", " ")
                frame_text = clip_descrip["frame_text"].replace("\n", " ")

                page_content = f"Audio: {audio_text}\nVisuals: {frame_text}"

                metadata = {
                    "video_name": video_descrip["name"],
                    "segment_id": video_descrip["index"],
                    "start_time": clip_descrip['start_time'],
                    "end_time": clip_descrip['end_time']
                }

                self.documents.append(Document(
                    page_content=page_content,
                    metadata=metadata
                ))
        self.qdrant_db = QdrantVectorStore.from_documents(
            self.documents,
            self.embeddings,
            location=":memory:",
            collection_name="video_rag_collection"
        )
    def chat(self, query):
        metadata_filter = models.Filter()
        retrieve = self.qdrant_db.similarity_search(
            query=query,
            k=3,
            filter=metadata_filter
        )



        prompt = self.build_automated_prompt(retrieve, query)

        response = chat(
            model="mistral",
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'num_ctx': 2048,  # Giới hạn context window để tiết kiệm RAM/VRAM
                'temperature': 0.1  # Giữ cho câu trả lời chính xác, không lan man
            }
        )
        return response['message']['content']

    def build_automated_prompt(self, retrieved_data, user_query):
        prompt = (
            "You are an intelligent AI assistant specializing in video analysis. "
            "Below is the extracted information (Audio and Visuals) from various video clips.\n\n"
            "=== [CONTEXT] ===\n"
        )

        if isinstance(retrieved_data, dict):
            results = retrieved_data.get("results", [])
        elif isinstance(retrieved_data, list):
            results = retrieved_data
        else:
            results = []

        for idx, item in enumerate(results):
            if not isinstance(item, dict):
                if hasattr(item, "dict"):
                    item = item.dict()
                elif hasattr(item, "model_dump"):
                    item = item.model_dump()
                else:
                    try:
                        item = vars(item)
                    except TypeError:
                        item = {}

            content = item.get("page_content", item.get("content", ""))

            meta = item.get("metadata", {})
            if not isinstance(meta, dict):
                if hasattr(meta, "dict"):
                    meta = meta.dict()
                elif hasattr(meta, "model_dump"):
                    meta = meta.model_dump()
                else:
                    try:
                        meta = vars(meta)
                    except TypeError:
                        meta = {}

            video_name = meta.get("video_name", "Unknown")
            start_time = meta.get("start_time", "N/A")
            end_time = meta.get("end_time", "N/A")

            # prompt += f"--- VIDEO CLIP {idx + 1} (File: {video_name} | From {start_time} to {end_time}) ---\n"
            prompt += f"{content}\n\n"

        prompt += "=== [TASK] ===\n"
        prompt += f"{user_query}\n\n"

        prompt += "=== [RULES] ===\n"
        prompt += "1. ONLY use the information provided in the [CONTEXT] section above. Do not use outside knowledge.\n"
        prompt += "2. Present your response clearly, using bullet points or short paragraphs for readability.\n"
        prompt += "3. If the answer cannot be found in the provided context, explicitly state: 'I cannot find this information in the provided context.'\n"

        return prompt

if __name__ == "__main__":
    agent = Agent()
    while True:
        query = input("Nhập câu hỏi của bạn (hoặc 'exit' để thoát): ")
        if query.lower() == "exit":
            break
        results = agent.chat(query)
        print(results)
        # for res in results:
        #     print("Nội dung text:", res.page_content)
        #     print("Metadata đính kèm:", res.metadata)
        #     print("--------------")

# According to the video, at what speed does the Earth spin, and what would happen to the atmosphere if the Earth suddenly stopped?
# According to the video clip 1, the Earth spins at a speed of 1000 miles per hour. If the Earth suddenly stopped spinning, the atmosphere would still be in motion and everything on the Earth's surface would fly into the atmosphere due to the momentum of the atmosphere. The water would migrate towards the poles as there would be no centrifugal force generating the huge bulge around the equator. This would leave behind a giant land mass at the equator. The Earth would also experience six months of day followed by six months of night, and without rotation, there would be no magnetic field to protect us from harmful solar wind, making it extremely difficult to survive.
