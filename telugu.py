import re
from collections import OrderedDict

class BrailleToTelugu:
    def __init__(self):
        self.initialize_mappings()
        
    def initialize_mappings(self):
        """Initialize mappings using your exact Braille representations"""
        # Consonants (Telugu to Braille as you specified)
        self.consonants = {
            'క': '⠅', 'ఖ': '⠨', 'గ': '⠛', 'ఘ': '⠣', 'ఙ': '⠬',
            'చ': '⠉', 'ఛ': '⠡', 'జ': '⠚', 'ఝ': '⠴', 'ఞ': '⠒',
            'ట': '⠾', 'ఠ': '⠺', 'డ': '⠫', 'ఢ': '⠿', 'ణ': '⠼',
            'త': '⠞', 'థ': '⠹', 'ద': '⠙', 'ధ': '⠮', 'న': '⠝',
            'ప': '⠏', 'ఫ': '⠖', 'బ': '⠃', 'భ': '⠘', 'మ': '⠍',
            'య': '⠽', 'ర': '⠗', 'ల': '⠇', 'వ': '⠧', 'శ': '⠩',
            'ష': '⠯', 'స': '⠎', 'హ': '⠓', 'ళ': '⠸'
        }
        
        # Vowel signs (as you specified)
        self.vowel_signs = {
            'ా': '⠜', 'ి': '⠊', 'ీ': '⠔', 'ు': '⠥', 'ూ': '⠳',
            'ృ': '⠐⠗', 'ౄ': '⠻', 'ౢ': '⠼', 'ౣ': '⠾', 'ె': '⠢',
            'ే': '⠑', 'ై': '⠌', 'ొ': '⠭', 'ో': '⠕', 'ౌ': '⠪'
        }
        
        # Vowels (as you specified)
        self.vowels = {
            'అ': '⠁', 'ఆ': '⠜', 'ఇ': '⠊', 'ఈ': '⠔', 'ఉ': '⠥',
            'ఊ': '⠳', 'ఋ': '⠺', 'ౠ': '⠻', 'ఌ': '⠼', 'ౡ': '⠾',
            'ఎ': '⠢', 'ఏ': '⠑', 'ఐ': '⠌', 'ఒ': '⠭', 'ఓ': '⠕',
            'ఔ': '⠪'
        }
        
        # Special characters
        self.special_chars = {
            'ం': '⠠', 'ః': '⠄', '్': '⠈', 'ఁ': '⠐'
        }
        
        # Create reverse mappings (Braille to Telugu)
        self.braille_map = {}
        self.vowel_signs_reverse = {v: k for k, v in self.vowel_signs.items()}
        
        # Add consonants first (priority for consonants)
        for telugu, braille in self.consonants.items():
            self.braille_map[braille] = telugu
            
        # Add vowels (only if not already in map)
        for telugu, braille in self.vowels.items():
            if braille not in self.braille_map:
                self.braille_map[braille] = telugu
                
        # Add special characters
        for telugu, braille in self.special_chars.items():
            self.braille_map[braille] = telugu
            
        # Multi-cell patterns (for conjuncts)
        self.multi_cell = OrderedDict([
            ('⠅⠈⠯', 'క్ష'), ('⠞⠈⠗', 'త్ర'), ('⠝⠈⠽', 'న్య'),
            ('⠍⠈⠽', 'మ్య'), ('⠇⠈⠇', 'ల్ల'), ('⠙⠈⠙', 'ద్ద'),
            ('⠏⠈⠏', 'ప్ప'), ('⠗⠈⠗', 'ర్ర'), ('⠐⠗', 'ృ')
        ])

    def convert_to_telugu(self, braille_text):
        """Convert Braille to Telugu using your exact representations"""
        telugu = []
        i = 0
        n = len(braille_text)
        
        while i < n:
            matched = False
            
            # 1. Check multi-cell patterns first
            for pattern, text in self.multi_cell.items():
                if braille_text.startswith(pattern, i):
                    telugu.append(text)
                    i += len(pattern)
                    matched = True
                    break
            
            # 2. Handle consonant + virama + consonant (generic ottu)
            if not matched and i+2 < n and braille_text[i+1] == '⠈':
                c1 = self.braille_map.get(braille_text[i], '')
                c2 = self.braille_map.get(braille_text[i+2], '')
                if c1 and c2:
                    telugu.append(f"{c1}్{c2}")
                    i += 3
                    matched = True
            
            # 3. Handle consonant + vowel sign combinations
            if not matched and i+1 < n:
                consonant = None
                # First try to get as consonant
                if braille_text[i] in self.braille_map:
                    consonant = self.braille_map[braille_text[i]]
                    # Verify it's actually a consonant
                    if consonant not in self.consonants:
                        consonant = None
                
                if consonant and braille_text[i+1] in self.vowel_signs_reverse:
                    vowel_sign = self.vowel_signs_reverse[braille_text[i+1]]
                    telugu.append(consonant + vowel_sign)
                    i += 2
                    matched = True
            
            # 4. Handle standalone characters
            if not matched:
                char = braille_text[i]
                if char in self.braille_map:
                    telugu.append(self.braille_map[char])
                elif char == ' ':
                    telugu.append(' ')
                i += 1
        
        return ''.join(telugu)

def process_uploaded_braille(uploaded_text):
    """Process uploaded Braille text"""
    converter = BrailleToTelugu()
    return converter.convert_to_telugu(uploaded_text)

# Example usage with uploaded text
if __name__ == "__main__":
    uploaded_braille = "⠅⠧⠊⠞"  # Should now convert to "ఎమిటి"
    
    telugu_output = process_uploaded_braille(uploaded_braille)
    print("Uploaded Braille:", uploaded_braille)
    print("Telugu Output:", telugu_output)


