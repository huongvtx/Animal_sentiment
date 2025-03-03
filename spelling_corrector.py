# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:03:15 2024

@author: vutxu
"""

# spell_corrector.py
import emoji
import pkg_resources
from textblob import TextBlob
from symspellpy import SymSpell, Verbosity

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Initialize SymSpell
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def correct_spelling_batch(texts):
    corrected_texts = []
    for text in texts:
        if not isinstance(text, str):
            corrected_texts.append('')
            continue
        
        words = text.split()
        corrected_words = []
        for word in words:
            if word in emoji.EMOJI_DATA:
                corrected_words.append(word)
            else:
                suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                if suggestions:
                    corrected_words.append(suggestions[0].term)
                else:
                    corrected_words.append(str(TextBlob(word).correct()))
        corrected_texts.append(' '.join(corrected_words))
    return corrected_texts
