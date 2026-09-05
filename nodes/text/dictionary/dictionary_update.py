"""Merge dictionaries into one."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import DICT

#: Input ids in socket order.
SLOTS = tuple(f"dictionary_{letter}" for letter in "abcdefghijklmnopqrstuvwx")


class DictionaryUpdate(io.ComfyNode):
    """Merge dictionaries, later inputs winning on a shared key."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Dictionary Update",
            display_name="Text Dictionary Update",
            search_aliases=["Text Dictionary Update", "dictionary merge", "dict update"],
            category="WAS Suite/Text/Dictionary",
            description=(
                "Merge two to 24 dictionaries into one. Keys present in more than one "
                "input take the value of the last input that carries them."
            ),
            inputs=[
                DICT.Input(
                    "dictionary_a",
                    tooltip=(
                        "The base dictionary. Its entries are the ones overwritten when a "
                        "later input carries the same key."
                    ),
                ),
                DICT.Input(
                    "dictionary_b",
                    tooltip=(
                        "Merged over dictionary_a, so a key in both takes this one's value."
                    ),
                ),
                DICT.Input(
                    "dictionary_c",
                    optional=True,
                    tooltip=(
                        "Merged over the first two. Unconnected, it contributes nothing."
                    ),
                ),
                DICT.Input(
                    "dictionary_d",
                    optional=True,
                    tooltip=(
                        "Merged last, so it wins every clash. Unconnected, it contributes "
                        "nothing."
                    ),
                ),
                DICT.Input(
                    "dictionary_e",
                    optional=True,
                    tooltip=(
                        "Dictionary 5. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_f",
                    optional=True,
                    tooltip=(
                        "Dictionary 6. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_g",
                    optional=True,
                    tooltip=(
                        "Dictionary 7. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_h",
                    optional=True,
                    tooltip=(
                        "Dictionary 8. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_i",
                    optional=True,
                    tooltip=(
                        "Dictionary 9. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_j",
                    optional=True,
                    tooltip=(
                        "Dictionary 10. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_k",
                    optional=True,
                    tooltip=(
                        "Dictionary 11. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_l",
                    optional=True,
                    tooltip=(
                        "Dictionary 12. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_m",
                    optional=True,
                    tooltip=(
                        "Dictionary 13. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_n",
                    optional=True,
                    tooltip=(
                        "Dictionary 14. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_o",
                    optional=True,
                    tooltip=(
                        "Dictionary 15. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_p",
                    optional=True,
                    tooltip=(
                        "Dictionary 16. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_q",
                    optional=True,
                    tooltip=(
                        "Dictionary 17. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_r",
                    optional=True,
                    tooltip=(
                        "Dictionary 18. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_s",
                    optional=True,
                    tooltip=(
                        "Dictionary 19. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_t",
                    optional=True,
                    tooltip=(
                        "Dictionary 20. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_u",
                    optional=True,
                    tooltip=(
                        "Dictionary 21. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_v",
                    optional=True,
                    tooltip=(
                        "Dictionary 22. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_w",
                    optional=True,
                    tooltip=(
                        "Dictionary 23. Its keys win over every input before it and lose to every one after."
                    ),
                ),
                DICT.Input(
                    "dictionary_x",
                    optional=True,
                    tooltip=(
                        "Dictionary 24. Its keys win over every input before it and lose to every one after."
                    ),
                ),
            ],
            outputs=[
                DICT.Output(
                    tooltip=(
                        "A new dictionary holding every entry of the connected inputs. The "
                        "inputs themselves are left alone."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        dictionary_a,
        dictionary_b,
        **extra,
    ) -> io.NodeOutput:
        merged = {**dictionary_a, **dictionary_b}
        for name in SLOTS[2:]:
            following = extra.get(name)
            if following is not None:
                merged = {**merged, **following}
        return io.NodeOutput(merged)
