"""
Agricultural Text Prompt Generator for SAM3

Generates context-rich text prompts for vision-language grounding in agricultural
segmentation tasks. Prompts incorporate domain-specific vocabulary and attributes.

Key Features:
- Attribute-based prompt generation (ripeness, health, species, etc.)
- Domain-specific templates for different agricultural contexts
- Negative prompt support for improved grounding
- Compositional prompts for complex concepts

Examples:
    "A ripe red apple fruit with healthy appearance"
    "Green unripe tomato with early growth stage"
    "Diseased wheat leaf showing rust infection"
    "Aphid pest on corn plant causing damage"
"""

from typing import Dict, List, Optional, Set
import random


class AgriculturalPromptGenerator:
    """
    Generate agricultural text prompts for SAM3 vision-language grounding
    
    Creates descriptive text prompts from category names and agricultural attributes.
    Supports multiple agricultural domains and concepts.
    """
    
    # Domain-specific templates
    TEMPLATES = {
        'fruit_ripeness': [
            "A {ripeness} {color} {category} {fruit_descriptor}",
            "{ripeness} {category} fruit {state_descriptor}",
            "{category} at {ripeness} stage with {color} color",
        ],
        'crop_weed': [
            "{category} plant growing in agricultural field",
            "{category} {growth_stage} vegetation",
            "Instance of {category} among crops",
        ],
        'disease': [
            "{health} {category} showing {disease} symptoms",
            "{category} leaf with {severity} {disease} infection",
            "Diseased {category} exhibiting {disease}",
        ],
        'pest': [
            "{category} pest on {host_plant} plant",
            "{category} insect causing {damage_type} damage",
            "{category} infestation at {severity} level",
        ],
        'multi_crop': [
            "{category} crop in {growth_stage} phase",
            "Field boundary of {category} region",
            "{category} agricultural land use",
        ],
        'greenhouse': [
            "{category} in controlled greenhouse environment",
            "Indoor cultivation of {category}",
            "{category} under greenhouse conditions",
        ],
        'general': [
            "An agricultural {category}",
            "{category} in field setting",
            "Instance of {category}",
        ]
    }
    
    # Attribute vocabularies
    RIPENESS_VOCAB = {
        'unripe': ['unripe', 'green', 'immature', 'early stage'],
        'ripe': ['ripe', 'mature', 'ready for harvest', 'fully developed'],
        'overripe': ['overripe', 'past maturity', 'late stage', 'deteriorating']
    }
    
    HEALTH_VOCAB = {
        'healthy': ['healthy', 'vigorous', 'normal', 'disease-free'],
        'diseased': ['diseased', 'infected', 'unhealthy', 'symptomatic'],
        'damaged': ['damaged', 'injured', 'stressed', 'compromised']
    }
    
    GROWTH_STAGE_VOCAB = {
        'seedling': ['seedling', 'germination', 'early vegetative'],
        'vegetative': ['vegetative growth', 'leaf development', 'biomass accumulation'],
        'flowering': ['flowering', 'reproductive', 'bloom stage'],
        'fruiting': ['fruiting', 'fruit development', 'maturation']
    }
    
    COLOR_VOCAB = {
        'green': ['green', 'greenish'],
        'yellow': ['yellow', 'golden', 'yellowish'],
        'red': ['red', 'reddish', 'crimson'],
        'brown': ['brown', 'brownish', 'tan'],
        'white': ['white', 'pale', 'light-colored']
    }
    
    SEVERITY_VOCAB = {
        'mild': ['mild', 'slight', 'minor'],
        'moderate': ['moderate', 'noticeable', 'significant'],
        'severe': ['severe', 'extensive', 'critical']
    }
    
    def __init__(self, concepts: List[str], domain: str = 'general'):
        """
        Initialize prompt generator
        
        Args:
            concepts: List of agricultural concepts to support
            domain: Primary agricultural domain (fruit_ripeness, crop_weed, etc.)
        """
        self.concepts = set(concepts) if concepts else set()
        self.domain = domain
        
        # Cache for generated prompts to ensure consistency
        self.prompt_cache = {}
    
    def generate_prompt(self, category: str, attributes: Dict[str, str] = None,
                       use_negative: bool = False) -> str:
        """
        Generate text prompt for agricultural object
        
        Args:
            category: Object category (e.g., 'apple', 'wheat', 'aphid')
            attributes: Agricultural attributes dictionary with keys like:
                - ripeness: 'unripe', 'ripe', 'overripe'
                - health: 'healthy', 'diseased', 'damaged'
                - growth_stage: 'seedling', 'vegetative', 'flowering', 'fruiting'
                - color: 'green', 'red', 'yellow', etc.
                - disease: specific disease name
                - variety: crop variety name
                - damage_type: 'leaf_damage', 'root_damage', etc.
                - severity: 'mild', 'moderate', 'severe'
            use_negative: Whether to generate negative prompt
        
        Returns:
            Descriptive text prompt string
        """
        attributes = attributes or {}
        
        # Create cache key
        cache_key = (category, tuple(sorted(attributes.items())), use_negative)
        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]
        
        # Generate prompt based on domain and attributes
        if use_negative:
            prompt = self._generate_negative_prompt(category, attributes)
        else:
            prompt = self._generate_positive_prompt(category, attributes)
        
        # Cache and return
        self.prompt_cache[cache_key] = prompt
        return prompt
    
    def _generate_positive_prompt(self, category: str, attributes: Dict) -> str:
        """Generate positive (object description) prompt"""
        # Get templates for domain
        templates = self.TEMPLATES.get(self.domain, self.TEMPLATES['general'])
        
        # Build attribute dictionary for template filling
        prompt_vars = {'category': category}
        
        # Add ripeness information
        if 'ripeness' in attributes:
            ripeness = attributes['ripeness']
            prompt_vars['ripeness'] = self._get_vocab_term(self.RIPENESS_VOCAB, ripeness)
            prompt_vars['fruit_descriptor'] = self._get_fruit_descriptor(ripeness)
            prompt_vars['state_descriptor'] = self._get_state_descriptor(ripeness)
        
        # Add health information
        if 'health' in attributes:
            health = attributes['health']
            prompt_vars['health'] = self._get_vocab_term(self.HEALTH_VOCAB, health)
        
        # Add growth stage
        if 'growth_stage' in attributes:
            stage = attributes['growth_stage']
            prompt_vars['growth_stage'] = self._get_vocab_term(
                self.GROWTH_STAGE_VOCAB, stage
            )
        
        # Add color
        if 'color' in attributes:
            color = attributes['color']
            prompt_vars['color'] = self._get_vocab_term(self.COLOR_VOCAB, color)
        
        # Add disease information
        if 'disease' in attributes:
            prompt_vars['disease'] = attributes['disease']
        
        # Add severity
        if 'severity' in attributes:
            severity = attributes['severity']
            prompt_vars['severity'] = self._get_vocab_term(self.SEVERITY_VOCAB, severity)
        
        # Add damage information
        if 'damage_type' in attributes:
            prompt_vars['damage_type'] = attributes['damage_type'].replace('_', ' ')
        
        # Add host plant for pests
        if 'host_plant' in attributes:
            prompt_vars['host_plant'] = attributes['host_plant']
        
        # Select template
        template = self._select_template(templates, prompt_vars)
        
        # Fill template
        try:
            prompt = template.format(**prompt_vars)
        except KeyError:
            # Fallback if template variables missing
            prompt = f"{category}"
            if 'ripeness' in prompt_vars:
                prompt = f"{prompt_vars['ripeness']} {prompt}"
            if 'health' in prompt_vars and prompt_vars['health'] != 'healthy':
                prompt = f"{prompt_vars['health']} {prompt}"
        
        return prompt
    
    def _generate_negative_prompt(self, category: str, attributes: Dict) -> str:
        """Generate negative (what to avoid) prompt"""
        negatives = []
        
        # Opposite ripeness
        if 'ripeness' in attributes:
            ripeness = attributes['ripeness']
            if ripeness == 'ripe':
                negatives.append('unripe or overripe')
            elif ripeness == 'unripe':
                negatives.append('ripe or overripe')
            else:
                negatives.append('unripe or ripe')
        
        # Opposite health
        if 'health' in attributes:
            health = attributes['health']
            if health == 'healthy':
                negatives.append('diseased or damaged')
            else:
                negatives.append('healthy')
        
        # Background objects
        negatives.extend(['background', 'soil', 'sky', 'equipment'])
        
        if negatives:
            return f"Not {', not '.join(negatives)}"
        return "background"
    
    def _select_template(self, templates: List[str], variables: Dict) -> str:
        """Select appropriate template based on available variables"""
        valid_templates = []
        
        for template in templates:
            # Check if all template variables are available
            required_vars = self._extract_template_vars(template)
            if all(var in variables for var in required_vars):
                valid_templates.append(template)
        
        if valid_templates:
            return random.choice(valid_templates)
        
        # Fallback to simplest template
        return "{category}"
    
    def _extract_template_vars(self, template: str) -> Set[str]:
        """Extract variable names from template string"""
        import re
        return set(re.findall(r'\{(\w+)\}', template))
    
    def _get_vocab_term(self, vocab: Dict[str, List[str]], key: str) -> str:
        """Get vocabulary term with variation"""
        if key in vocab:
            return random.choice(vocab[key])
        return key
    
    def _get_fruit_descriptor(self, ripeness: str) -> str:
        """Get fruit descriptor based on ripeness"""
        descriptors = {
            'unripe': 'not ready for consumption',
            'ripe': 'ready for harvest',
            'overripe': 'past optimal harvest time'
        }
        return descriptors.get(ripeness, '')
    
    def _get_state_descriptor(self, ripeness: str) -> str:
        """Get state descriptor"""
        descriptors = {
            'unripe': 'in early maturation',
            'ripe': 'at peak maturity',
            'overripe': 'beyond peak maturity'
        }
        return descriptors.get(ripeness, '')
    
    def generate_batch_prompts(self, categories: List[str], 
                               attributes_list: List[Dict]) -> List[str]:
        """
        Generate prompts for batch of objects
        
        Args:
            categories: List of category names
            attributes_list: List of attribute dictionaries
        
        Returns:
            List of text prompts
        """
        prompts = []
        for category, attributes in zip(categories, attributes_list):
            prompt = self.generate_prompt(category, attributes)
            prompts.append(prompt)
        return prompts
    
    def get_concept_prompts(self, concept: str) -> List[str]:
        """
        Get pre-defined prompts for specific agricultural concept
        
        Args:
            concept: Concept name (e.g., 'ripeness', 'disease', 'growth_stage')
        
        Returns:
            List of example prompts for the concept
        """
        concept_prompts = {
            'ripeness': [
                "unripe green fruit",
                "ripe mature fruit ready for harvest",
                "overripe fruit past optimal harvest"
            ],
            'disease': [
                "healthy plant with no disease symptoms",
                "diseased plant showing infection",
                "severely infected plant with extensive damage"
            ],
            'growth_stage': [
                "seedling in early growth",
                "vegetative plant with leaf development",
                "flowering plant in reproductive stage",
                "fruiting plant with developing fruits"
            ],
            'health': [
                "healthy vigorous plant",
                "stressed plant with damage",
                "diseased plant with symptoms"
            ],
            'weed': [
                "crop plant in agricultural field",
                "weed species among crops",
                "invasive weed requiring removal"
            ]
        }
        return concept_prompts.get(concept, [f"agricultural {concept}"])
    
    def clear_cache(self):
        """Clear prompt cache"""
        self.prompt_cache.clear()


class CompositePromptGenerator:
    """
    Generate composite prompts that combine multiple agricultural concepts
    
    Useful for complex segmentation tasks requiring multi-attribute reasoning.
    """
    
    def __init__(self, base_generator: AgriculturalPromptGenerator):
        """
        Initialize composite generator
        
        Args:
            base_generator: Base agricultural prompt generator
        """
        self.base_generator = base_generator
    
    def generate_composite_prompt(self, category: str, 
                                  primary_attributes: Dict,
                                  secondary_attributes: Dict = None) -> str:
        """
        Generate composite prompt from multiple attribute sets
        
        Args:
            category: Object category
            primary_attributes: Primary attributes (e.g., ripeness, health)
            secondary_attributes: Secondary attributes (e.g., location, context)
        
        Returns:
            Composite prompt string
        """
        # Generate base prompt
        base_prompt = self.base_generator.generate_prompt(category, primary_attributes)
        
        # Add secondary context if provided
        if secondary_attributes:
            context_parts = []
            
            if 'location' in secondary_attributes:
                context_parts.append(f"in {secondary_attributes['location']}")
            
            if 'lighting' in secondary_attributes:
                context_parts.append(f"under {secondary_attributes['lighting']} lighting")
            
            if 'density' in secondary_attributes:
                context_parts.append(f"with {secondary_attributes['density']} density")
            
            if context_parts:
                base_prompt = f"{base_prompt} {' '.join(context_parts)}"
        
        return base_prompt
    
    def generate_relationship_prompt(self, category1: str, category2: str,
                                    relationship: str) -> str:
        """
        Generate prompt describing relationship between objects
        
        Args:
            category1: First object category
            category2: Second object category
            relationship: Relationship type ('adjacent', 'touching', 'overlapping')
        
        Returns:
            Relationship prompt
        """
        templates = {
            'adjacent': f"{category1} adjacent to {category2}",
            'touching': f"{category1} in contact with {category2}",
            'overlapping': f"{category1} overlapping with {category2}",
            'near': f"{category1} near {category2}"
        }
        return templates.get(relationship, f"{category1} and {category2}")


def create_prompt_generator(experiment_type: str, concepts: List[str]) -> AgriculturalPromptGenerator:
    """
    Factory function to create prompt generator for specific experiment
    
    Args:
        experiment_type: Experiment type identifier
        concepts: List of concepts for the experiment
    
    Returns:
        Configured prompt generator
    """
    domain_mapping = {
        'exp01_fruit_ripeness': 'fruit_ripeness',
        'exp02_crop_weed': 'crop_weed',
        'exp03_disease': 'disease',
        'exp04_multi_crop': 'multi_crop',
        'exp05_pests': 'pest',
        'exp06_greenhouse': 'greenhouse'
    }
    
    domain = domain_mapping.get(experiment_type, 'general')
    return AgriculturalPromptGenerator(concepts=concepts, domain=domain)
