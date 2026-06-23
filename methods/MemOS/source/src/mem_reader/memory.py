from datetime import datetime
from typing import Any

from memos.llms.base import BaseLLM


class Memory:
    """Class representing the memory structure for storing and organizing memory content."""

    def __init__(
        self,
        user_id: str,
        session_id: str,
        created_at: datetime,
    ):
        """
        Initialize the Memory structure.

        Args:
            user_id: User identifier
            session_id: Session identifier
            created_at: Creation timestamp
        """
        self.objective_memory: dict[str, dict[str, Any]] = {}
        self.subjective_memory: dict[str, dict[str, Any]] = {}
        self.scene_memory = {
            "qa_pair": {
                "section": [],
                "info": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "created_at": created_at,
                    "summary": "",
                    "label": [],
                },
            },
            "document": {
                "section": [],
                "info": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "created_at": created_at,
                    "doc_type": "",  # pdf, txt, etc.
                    "doc_category": "",  # research_paper, news, etc.
                    "doc_name": "",
                    "summary": "",
                    "label": [],
                },
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Memory object to a dictionary.

        Returns:
            Dictionary representation of the Memory object
        """
        return {
            "objective_memory": self.objective_memory,
            "subjective_memory": self.subjective_memory,
            "scene_memory": self.scene_memory,
        }

    def update_user_memory(
        self,
        memory_type: str,
        key: str,
        value: Any,
        origin_data: str,
        confidence_score: float = 1.0,
        timestamp: str | None = None,
    ) -> None:
        """
        Update a memory item in either objective_memory or subjective_memory.
        If a key already exists, the new memory item's info will replace the existing one,
        and the values will be connected.

        Args:
            memory_type: Type of memory to update ('objective' or 'subjective')
            key: Key for the memory item. Must be one of:

                | Memory Type       | Key                  | Description                                             |
                |-------------------|----------------------|---------------------------------------------------------|
                | objective_memory  | nickname             | User's preferred name or alias                          |
                | objective_memory  | gender               | User's gender (male, female, other)                     |
                | objective_memory  | personality          | User's personality traits or MBTI type                  |
                | objective_memory  | birth                | User's birthdate or age information                     |
                | objective_memory  | education            | User's educational background                           |
                | objective_memory  | work                 | User's professional history                             |
                | objective_memory  | achievement          | User's notable accomplishments                          |
                | objective_memory  | occupation           | User's current job or role                              |
                | objective_memory  | residence            | User's home location or living situation                |
                | objective_memory  | location             | User's current geographical location                    |
                | objective_memory  | income               | User's financial information                            |
                | objective_memory  | preference           | User's likes and dislikes                               |
                | objective_memory  | expertise            | User's skills and knowledge areas                       |
                | objective_memory  | language             | User's language proficiency                             |
                | objective_memory  | hobby                | User's recreational activities                          |
                | objective_memory  | goal                 | User's long-term aspirations                            |
                |-------------------|----------------------|---------------------------------------------------------|
                | subjective_memory | current_mood         | User's current emotional state                          |
                | subjective_memory | response_style       | User's preferred interaction style                      |
                | subjective_memory | language_style       | User's language patterns and preferences                |
                | subjective_memory | information_density  | User's preference for detail level in responses         |
                | subjective_memory | interaction_pace     | User's preferred conversation speed and frequency       |
                | subjective_memory | followed_topic       | Topics the user is currently interested in              |
                | subjective_memory | current_goal         | User's immediate objectives in the conversation         |
                | subjective_memory | content_type         | User's preferred field of interest (e.g., technology, finance, etc.)               |
                | subjective_memory | role_preference      | User's preferred assistant role (e.g., domain expert, translation assistant, etc.) |

            value: Value to store
            origin_data: Original data that led to this memory
            confidence_score: Confidence score (0.0 to 1.0)
            timestamp: Timestamp string, if None current time will be used
        """
        if timestamp is None:
            timestamp = datetime.now()

        memory_item = {
            "value": value,
            "info": {
                "timestamp": timestamp,
                "confidence_score": confidence_score,
                "origin_data": origin_data,
            },
        }

        if memory_type == "objective":
            memory_dict = self.objective_memory
        elif memory_type == "subjective":
            memory_dict = self.subjective_memory
        else:
            raise ValueError(
                f"Invalid memory_type: {memory_type}. Must be 'objective' or 'subjective'."
            )

        # Check if key already exists
        if key in memory_dict:
            existing_item = memory_dict[key]

            # Connect the values (keep history but present as a connected string)
            combined_value = f"{existing_item['value']} | {value}"

            # Update the memory item with combined value and new info (using the newest info)
            memory_dict[key] = {
                "value": combined_value,
                "info": memory_item["info"],  # Use the new info
            }
        else:
            # If key doesn't exist, simply add the new memory item
            memory_dict[key] = memory_item

    def add_qa_batch(
        self, batch_summary: str, pair_summaries: list[dict], themes: list[str], order: int
    ) -> None:
        """
        Add a batch of Q&A pairs to the scene memory as a single subsection.

        Args:
            batch_summary: The summary of the entire batch
            pair_summaries: List of dictionaries, each containing:
                - question: The summarized question for a single pair
                - summary: The original dialogue for a single pair
                - prompt: The prompt used for summarization
                - time: The extracted time information (if any)
            themes: List of themes associated with the batch
            order: Order of the batch in the sequence
        """
        qa_subsection = {
            "subsection": {},
            "info": {
                "summary": batch_summary,
                "label": themes,
                "origin_data": "",
                "order": order,
            },
        }

        for pair in pair_summaries:
            qa_subsection["subsection"][pair["question"]] = {
                "summary": pair["summary"],
                "sources": pair["prompt"].split("\n\n", 1)[-1],
                "time": pair.get("time", ""),  # Add time field with default empty string
            }

        self.scene_memory["qa_pair"]["section"].append(qa_subsection)

    def add_document_chunk_group(
        self, summary: str, label: list[str], order: int, sub_chunks: list
    ) -> None:
        """
        Add a group of document chunks as a single section with multiple facts in the subsection.

        Args:
            summary: The summary of the large chunk
            label: List of theme labels for the large chunk
            order: Order of the large chunk in the sequence
            sub_chunks: List of dictionaries containing small chunks information,
                        each with keys: 'question', 'chunk_text', 'prompt'
        """
        doc_section = {
            "subsection": {},
            "info": {
                "summary": summary,
                "label": label,
                "origin_data": "",
                "order": order,
            },
        }

        # Add each small chunk as a fact in the subsection
        for sub_chunk in sub_chunks:
            question = sub_chunk["question"]
            doc_section["subsection"][question] = {
                "summary": sub_chunk["chunk_text"],
                "sources": sub_chunk["prompt"].split("\n\n", 1)[-1],
            }

        self.scene_memory["document"]["section"].append(doc_section)

    def process_qa_pair_summaries(self, llm: BaseLLM | None = None) -> None:
        """
        Process all qa_pair subsection summaries to generate a section summary.

        Args:
            llm: Optional LLM instance to generate summary. If None, concatenates subsection summaries.
        Returns:
            The generated section summary
        """
        all_summaries = []
        all_labels = set()

        # Collect all subsection summaries and labels
        for section in self.scene_memory["qa_pair"]["section"]:
            if "info" in section and "summary" in section["info"]:
                all_summaries.append(section["info"]["summary"])
            if "info" in section and "label" in section["info"]:
                all_labels.update(section["info"]["label"])

        # Generate summary
        if llm is not None:
            # Use LLM to generate a coherent summary
            all_summaries_str = "\n".join(all_summaries)
            messages = [
                {
                    "role": "user",
                    "content": f"Summarize this text into a concise and objective sentence that captures its main idea. Provide only the required content directly, without including any additional information.\n\n{all_summaries_str}",
                }
            ]
            section_summary = llm.generate(messages)
        else:
            # Simple concatenation of summaries
            section_summary = " ".join(all_summaries)

        # Update the section info
        self.scene_memory["qa_pair"]["info"]["summary"] = section_summary
        self.scene_memory["qa_pair"]["info"]["label"] = list(all_labels)

    def process_document_summaries(self, llm=None) -> str:
        """
        Process all document subsection summaries to generate a section summary.

        Args:
            llm: Optional LLM instance to generate summary. If None, concatenates subsection summaries.
        Returns:
            The generated section summary
        """
        all_summaries = []
        all_labels = set()

        # Collect all subsection summaries and labels
        for section in self.scene_memory["document"]["section"]:
            if "info" in section and "summary" in section["info"]:
                all_summaries.append(section["info"]["summary"])
            if "info" in section and "label" in section["info"]:
                all_labels.update(section["info"]["label"])

        # Generate summary
        if llm is not None:
            # Use LLM to generate a coherent summary
            all_summaries_str = "\n".join(all_summaries)
            messages = [
                {
                    "role": "user",
                    "content": f"Summarize this text into a concise and objective sentence that captures its main idea. Provide only the required content directly, without including any additional information.\n\n{all_summaries_str}",
                }
            ]
            section_summary = llm.generate(messages)
        else:
            # Simple concatenation of summaries
            section_summary = " ".join(all_summaries)

        # Update the section info
        self.scene_memory["document"]["info"]["summary"] = section_summary
        self.scene_memory["document"]["info"]["label"] = list(all_labels)

        return section_summary
