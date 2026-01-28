"""Template-based text generation for Sanskrit, Hindi, and English."""
from __future__ import annotations

import random
from typing import List, Dict, Any


class TemplateGenerator:
    """Generate text using templates for different languages and domains."""

    def __init__(self, language: str):
        self.language = language
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load templates for different domains."""
        if self.language == "sanskrit":
            return self._sanskrit_templates()
        elif self.language == "hindi":
            return self._hindi_templates()
        elif self.language == "english":
            return self._english_templates()
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    def _sanskrit_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Sanskrit templates across domains."""
        return {
            "general_knowledge": [
                {
                    "template": "विज्ञानस्य {subject} विषये {fact} एतत् महत्त्वपूर्णम् अस्ति।",
                    "vars": {
                        "subject": ["गणितम्", "भौतिकशास्त्रम्", "रसायनशास्त्रम्", "जीवविज्ञानम्"],
                        "fact": ["अणुः सूक्ष्मतमः कणः", "प्रकाशः तरङ्गरूपेण गच्छति", "जलस्य रासायनिकसूत्रं H2O अस्ति"]
                    }
                },
                {
                    "template": "{concept} इत्यस्य अर्थः {meaning} इति भवति।",
                    "vars": {
                        "concept": ["धर्मः", "कर्म", "योगः", "ध्यानम्"],
                        "meaning": ["कर्तव्यम्", "क्रिया", "समाधिः", "चिन्तनम्"]
                    }
                }
            ],
            "cultural": [
                {
                    "template": "भारतीयसंस्कृतौ {festival} महत्त्वपूर्णं पर्व अस्ति। अस्मिन् पर्वणि जनाः {activity} कुर्वन्ति।",
                    "vars": {
                        "festival": ["दीपावली", "होली", "दशहरा", "पोङ्गल्"],
                        "activity": ["दीपान् प्रज्वालयन्ति", "रङ्गैः क्रीडन्ति", "पूजां कुर्वन्ति", "उत्सवं आचरन्ति"]
                    }
                },
                {
                    "template": "वेदेषु {veda} इत्यस्य ज्ञानं {knowledge} विषये प्राप्यते।",
                    "vars": {
                        "veda": ["ऋग्वेदः", "यजुर्वेदः", "सामवेदः", "अथर्ववेदः"],
                        "knowledge": ["स्तोत्राणि", "यज्ञविधयः", "सङ्गीतम्", "आयुर्वेदः"]
                    }
                }
            ],
            "technical": [
                {
                    "template": "सङ्गणके {operation} इति क्रिया {description} कर्तुं उपयुज्यते।",
                    "vars": {
                        "operation": ["प्रोग्राम्-लेखनम्", "डेटा-संग्रहणम्", "गणना", "सञ्चारः"],
                        "description": ["निर्देशान् निर्मातुम्", "सूचनां स्थापयितुम्", "अङ्कानां परिगणनाय", "सन्देशान् प्रेषयितुम्"]
                    }
                }
            ],
            "conversational": [
                {
                    "template": "कथं भवान् {action} करोति? अहं {response} करोमि।",
                    "vars": {
                        "action": ["पठति", "लिखति", "गच्छति", "खादति"],
                        "response": ["सावधानेन", "शनैः", "त्वरया", "आनन्देन"]
                    }
                }
            ]
        }

    def _hindi_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Hindi templates across domains."""
        return {
            "general_knowledge": [
                {
                    "template": "{subject} विज्ञान में {fact} एक महत्वपूर्ण तथ्य है।",
                    "vars": {
                        "subject": ["गणित", "भौतिकी", "रसायन", "जीवविज्ञान", "खगोल"],
                        "fact": ["परमाणु सबसे छोटा कण है", "प्रकाश तरंग के रूप में चलता है", "पानी का रासायनिक सूत्र H2O है", "पृथ्वी सूर्य के चारों ओर घूमती है"]
                    }
                },
                {
                    "template": "इतिहास में {event} का {year} में महत्वपूर्ण योगदान रहा है।",
                    "vars": {
                        "event": ["स्वतंत्रता संग्राम", "औद्योगिक क्रांति", "वैज्ञानिक खोज", "सांस्कृतिक पुनर्जागरण"],
                        "year": ["1947", "19वीं सदी", "20वीं सदी", "आधुनिक युग"]
                    }
                }
            ],
            "cultural": [
                {
                    "template": "भारतीय संस्कृति में {element} का विशेष स्थान है। यह {significance} का प्रतीक है।",
                    "vars": {
                        "element": ["संगीत", "नृत्य", "साहित्य", "कला", "योग"],
                        "significance": ["आध्यात्मिकता", "सांस्कृतिक विविधता", "परंपरा", "ज्ञान"]
                    }
                },
                {
                    "template": "{festival} त्योहार में लोग {activity} करते हैं और {tradition} का पालन करते हैं।",
                    "vars": {
                        "festival": ["दिवाली", "होली", "ईद", "क्रिसमस", "गुरुपर्व"],
                        "activity": ["दीप जलाते", "रंग खेलते", "प्रार्थना करते", "उपहार देते"],
                        "tradition": ["परंपराओं", "रीति-रिवाजों", "मान्यताओं", "संस्कारों"]
                    }
                }
            ],
            "technical": [
                {
                    "template": "{technology} तकनीक का उपयोग {application} में किया जाता है। यह {benefit} में मदद करता है।",
                    "vars": {
                        "technology": ["कृत्रिम बुद्धिमत्ता", "मशीन लर्निंग", "ब्लॉकचेन", "क्वांटम कंप्यूटिंग"],
                        "application": ["स्वास्थ्य सेवा", "शिक्षा", "वित्त", "कृषि"],
                        "benefit": ["समस्या समाधान", "दक्षता बढ़ाने", "नवाचार", "विकास"]
                    }
                },
                {
                    "template": "प्रोग्रामिंग में {concept} का {use} के लिए उपयोग होता है।",
                    "vars": {
                        "concept": ["लूप", "फंक्शन", "वेरिएबल", "एल्गोरिथम"],
                        "use": ["पुनरावृत्ति", "कोड संगठन", "डेटा संग्रहण", "समस्या समाधान"]
                    }
                }
            ],
            "conversational": [
                {
                    "template": "आप {action} कैसे करते हैं? मैं {method} का उपयोग करता हूं।",
                    "vars": {
                        "action": ["सीखते", "काम करते", "योजना बनाते", "निर्णय लेते"],
                        "method": ["अभ्यास और धैर्य", "व्यवस्थित तरीके", "विश्लेषण और चिंतन", "अनुभव और ज्ञान"]
                    }
                },
                {
                    "template": "{topic} के बारे में {question} क्या है? {answer} यह है।",
                    "vars": {
                        "topic": ["विज्ञान", "इतिहास", "कला", "प्रौद्योगिकी"],
                        "question": ["सबसे महत्वपूर्ण बात", "मुख्य सिद्धांत", "आधारभूत तथ्य"],
                        "answer": ["अवलोकन और प्रयोग", "समय और संदर्भ", "अभिव्यक्ति और सौंदर्य", "नवाचार और उपयोगिता"]
                    }
                }
            ]
        }

    def _english_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """English templates across domains."""
        return {
            "general_knowledge": [
                {
                    "template": "In {subject}, {fact} is a fundamental principle that explains {application}.",
                    "vars": {
                        "subject": ["mathematics", "physics", "chemistry", "biology", "computer science"],
                        "fact": ["the law of conservation of energy", "Pythagoras theorem", "Newton's laws", "DNA structure", "algorithm complexity"],
                        "application": ["natural phenomena", "geometric relationships", "motion and force", "heredity", "computational efficiency"]
                    }
                },
                {
                    "template": "The concept of {concept} is essential in understanding {domain} because it {reason}.",
                    "vars": {
                        "concept": ["evolution", "gravity", "democracy", "sustainability", "artificial intelligence"],
                        "domain": ["biology", "astronomy", "political science", "environmental science", "technology"],
                        "reason": ["explains diversity of life", "governs planetary motion", "ensures fair governance", "protects our planet", "enhances human capabilities"]
                    }
                }
            ],
            "cultural": [
                {
                    "template": "Indian culture emphasizes {value} which is reflected in {practice} and represents {meaning}.",
                    "vars": {
                        "value": ["family bonds", "spiritual growth", "respect for elders", "unity in diversity", "knowledge seeking"],
                        "practice": ["joint family systems", "meditation and yoga", "traditional ceremonies", "festivals celebrating all religions", "guru-shishya tradition"],
                        "meaning": ["collective well-being", "inner peace", "cultural continuity", "social harmony", "wisdom transmission"]
                    }
                },
                {
                    "template": "The {art_form} tradition from India has {contribution} to world culture through {aspect}.",
                    "vars": {
                        "art_form": ["classical music", "dance", "literature", "architecture", "textile"],
                        "contribution": ["contributed significantly", "enriched", "influenced", "inspired"],
                        "aspect": ["its complex rhythmic patterns", "expressive storytelling", "philosophical depth", "innovative techniques", "intricate craftsmanship"]
                    }
                }
            ],
            "technical": [
                {
                    "template": "In software engineering, {concept} is implemented using {method} to achieve {goal}.",
                    "vars": {
                        "concept": ["abstraction", "encapsulation", "polymorphism", "concurrency", "scalability"],
                        "method": ["object-oriented design", "design patterns", "microservices", "asynchronous programming", "distributed systems"],
                        "goal": ["code reusability", "maintainability", "flexibility", "performance", "reliability"]
                    }
                },
                {
                    "template": "Machine learning models use {technique} for {task} which improves {outcome}.",
                    "vars": {
                        "technique": ["neural networks", "decision trees", "clustering", "reinforcement learning", "transfer learning"],
                        "task": ["classification", "prediction", "pattern recognition", "optimization", "feature extraction"],
                        "outcome": ["accuracy", "efficiency", "generalization", "decision making", "automation"]
                    }
                }
            ],
            "conversational": [
                {
                    "template": "How do you approach {task}? I typically {method} because it {benefit}.",
                    "vars": {
                        "task": ["problem solving", "learning new skills", "making decisions", "creative work"],
                        "method": ["break it into smaller steps", "practice regularly", "analyze different perspectives", "brainstorm and iterate"],
                        "benefit": ["makes it manageable", "builds proficiency", "leads to better choices", "encourages innovation"]
                    }
                },
                {
                    "template": "What is the best way to {action}? Consider {factor} and {approach}.",
                    "vars": {
                        "action": ["learn effectively", "work collaboratively", "manage time", "stay motivated"],
                        "factor": ["your goals", "available resources", "personal strengths", "current challenges"],
                        "approach": ["set clear objectives", "leverage teamwork", "prioritize tasks", "celebrate progress"]
                    }
                }
            ]
        }

    def generate(self, domain: str, count: int = 10) -> List[str]:
        """Generate text samples from templates."""
        if domain not in self.templates:
            raise ValueError(f"Domain {domain} not found for language {self.language}")

        samples = []
        domain_templates = self.templates[domain]

        for _ in range(count):
            template_data = random.choice(domain_templates)
            template = template_data["template"]
            vars_dict = template_data["vars"]

            # Fill template with random choices
            filled = template
            for var_name, choices in vars_dict.items():
                filled = filled.replace(f"{{{var_name}}}", random.choice(choices))

            samples.append(filled)

        return samples
