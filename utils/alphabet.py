"""
alphabet.py - Bidirectional Token-Index Mapping

Provides the Alphabet class which maintains a bidirectional mapping between
string tokens (words, characters, or labels) and integer indices. This is
essential for converting between human-readable text and the integer IDs
that neural networks operate on.

Key features:
- Automatic index assignment when adding new tokens
- Built-in PAD (index 0) and UNKNOWN tokens for word/char alphabets
- Freeze/unfreeze functionality to prevent new tokens during inference
- JSON serialization for saving/loading vocabularies
- Support for both Python 2 and Python 3

Index 0 is reserved as a default (padding) index. Real tokens start from index 1.
"""

from __future__ import print_function
import json
import os
import sys


class Alphabet:
    """
    Bidirectional mapping between string tokens and integer indices.
    
    Args:
        name: Identifier for this alphabet (e.g., 'word', 'character', 'label').
        label: If True, skip adding PAD/UNKNOWN tokens (labels don't need them).
        keep_growing: If True, automatically add unseen tokens. Set to False
                      after building the vocabulary to map unknowns to UNK.
    """
    def __init__(self, name, label=False, keep_growing=True):
        self.name = name
        self.UNKNOWN = "</unk>"  # Unknown token placeholder
        self.PAD = '<PAD>'      # Padding token placeholder
        self.label = label
        self.instance2index = {}  # Token string -> integer index
        self.instances = []       # Ordered list of token strings
        self.keep_growing = keep_growing

        # Index 0 is reserved as the default/padding index
        self.default_index = 0
        self.next_index = 1

        # For non-label alphabets, pre-populate with PAD and UNKNOWN tokens
        if not self.label:
            self.add(self.PAD)
            self.add(self.UNKNOWN)

    def clear(self, keep_growing=True):
        """Reset the alphabet to empty state."""
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        self.default_index = 0
        self.next_index = 1

    def add(self, instance):
        """Add a token to the alphabet if not already present."""
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        """
        Get the integer index for a token.
        
        If keep_growing=True and token is new, add it and return its new index.
        If keep_growing=False and token is unknown, return the UNKNOWN index.
        """
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                return self.instance2index[self.UNKNOWN]

    def get_instance(self, index):
        """
        Get the token string for a given integer index.
        Returns the first label/token if index is 0 or out of bounds.
        """
        if index == 0:
            if self.label:
                return self.instances[0]
            return self.instances[0]
        try:
            return self.instances[index - 1]
        except IndexError:
            print('WARNING:Alphabet get_instance ,unknown instance, return the first label.')
            return self.instances[1]

    def size(self):
        """Return the total number of entries (including the reserved index 0)."""
        return len(self.instances) + 1

    def iteritems(self):
        """Return iterator over (token, index) pairs. Handles Python 2/3 compatibility."""
        if sys.version_info[0] < 3:
            return self.instance2index.iteritems()
        else:
            return self.instance2index.items()

    def enumerate_items(self, start=1):
        """Enumerate alphabet items starting from a given index."""
        if start < 1 or start >= self.size():
            raise IndexError("Enumerate is allowed between [1 : size of the alphabet)")
        return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    def close(self):
        """Freeze the alphabet: stop adding new tokens (map unknowns to UNK)."""
        self.keep_growing = False

    def open(self):
        """Unfreeze the alphabet: allow adding new tokens again."""
        self.keep_growing = True

    def get_content(self):
        """Get alphabet contents as a serializable dictionary."""
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        """Load alphabet from a JSON dictionary."""
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        Save alphabet to a JSON file.
        
        Args:
            output_directory: Directory to save the JSON file.
            name: Optional custom filename (defaults to alphabet name).
        """
        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
        except Exception as e:
            print("Exception: Alphabet is not saved: " % repr(e))

    def load(self, input_directory, name=None):
        """
        Load alphabet from a JSON file.
        
        Args:
            input_directory: Directory containing the JSON file.
            name: Optional custom filename (defaults to alphabet name).
        """
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
