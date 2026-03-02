from app.config import config
from app.logger.logger import get_logger

logger = get_logger(__name__)

# Risk patterns — queries we handle carefully
RISK_PATTERNS = {
    "medical_advice": [
        "should i take", "can i take", "is it safe",
        "will i be covered if", "do i have", "am i eligible",
        "should i claim", "will my claim", "can i claim"
    ],
    "legal_advice": [
        "is it legal", "can i sue", "legal action",
        "consumer court", "complaint against",
        "policy is wrong", "they cheated"
    ],
    "personal_claim": [
        "my claim was rejected", "they denied my claim",
        "i was not reimbursed", "my hospital bill",
        "my surgery", "my treatment"
    ]
}

# Sensitive terms that need extra care
SENSITIVE_TERMS = [
    "cancer", "hiv", "aids", "mental health",
    "suicide", "pregnancy", "maternity",
    "pre-existing", "ped", "critical illness"
]


def check_risk(query: str) -> dict:
    """
    Analyze query for risk patterns.
    
    Returns:
        {
            "risk_level": "low" | "medium" | "high",
            "risk_type": None | "medical_advice" | "legal_advice" | "personal_claim",
            "sensitive_terms": [],
            "should_escalate": False,
            "escalation_reason": None
        }
    """
    query_lower = query.lower()
    
    # Check risk patterns
    detected_risk = None
    for risk_type, patterns in RISK_PATTERNS.items():
        for pattern in patterns:
            if pattern in query_lower:
                detected_risk = risk_type
                break
        if detected_risk:
            break
    
    # Check sensitive terms
    detected_sensitive = [
        term for term in SENSITIVE_TERMS
        if term in query_lower
    ]
    
    # Determine risk level
    if detected_risk in ["medical_advice", "legal_advice"]:
        risk_level = "high"
        should_escalate = True
        escalation_reason = f"Query requires {detected_risk.replace('_', ' ')} — beyond document scope"
    elif detected_risk == "personal_claim":
        risk_level = "medium"
        should_escalate = False
        escalation_reason = "Personal claim situation — answer from policy only"
    elif detected_sensitive:
        risk_level = "medium"
        should_escalate = False
        escalation_reason = f"Sensitive terms detected: {', '.join(detected_sensitive)}"
    else:
        risk_level = "low"
        should_escalate = False
        escalation_reason = None
    
    result = {
        "risk_level": risk_level,
        "risk_type": detected_risk,
        "sensitive_terms": detected_sensitive,
        "should_escalate": should_escalate,
        "escalation_reason": escalation_reason
    }
    
    logger.info(f"Risk check: level={risk_level} | type={detected_risk}")
    return result


def check_confidence(retrieval_results: list[dict]) -> dict:
    """
    Evaluate retrieval quality to decide if we can answer.
    
    Returns:
        {
            "confidence": "high" | "medium" | "low",
            "should_answer": True | False,
            "reason": "..."
        }
    """
    # No results at all
    if not retrieval_results:
        return {
            "confidence": "low",
            "should_answer": False,
            "reason": "No relevant chunks found in documents"
        }
    
    top_result = retrieval_results[0]
    top_score = top_result.get("hybrid_score", 0)
    signals = top_result.get("confidence_signals", {})
    
    # High confidence conditions
    if (top_score >= config.CONFIDENCE_THRESHOLD and
            signals.get("found_in_both", False)):
        return {
            "confidence": "high",
            "should_answer": True,
            "reason": "Strong semantic + keyword match found"
        }
    
    # Medium confidence
    if top_score >= config.CONFIDENCE_THRESHOLD:
        return {
            "confidence": "medium",
            "should_answer": True,
            "reason": "Semantic match found — answer with citation"
        }
    
    # Low confidence
    return {
        "confidence": "low",
        "should_answer": False,
        "reason": "Retrieval confidence below threshold — cannot find in documents"
    }


def evaluate_query(query: str,
                   retrieval_results: list[dict]) -> dict:
    """
    Full evaluation — combines risk check + confidence check.
    Single entry point for pipeline to call.
    
    Returns:
        {
            "should_answer": True | False,
            "risk": {...},
            "confidence": {...},
            "final_decision": "answer" | "escalate" | "not_found",
            "safe_message": None | "..."
        }
    """
    # Step 1: Risk check
    risk = check_risk(query)
    
    # Step 2: Confidence check
    confidence = check_confidence(retrieval_results)
    
    # Step 3: Final decision
    if risk["should_escalate"]:
        decision = "escalate"
        safe_message = (
            f"I can provide general policy information, but this query "
            f"involves {risk['risk_type'].replace('_', ' ')} which requires "
            f"professional guidance. Please consult your insurance advisor "
            f"or contact the insurer directly. "
            f"Here is what the policy document states: "
        )
    elif not confidence["should_answer"]:
        decision = "not_found"
        safe_message = (
            "I couldn't find specific information about this in the "
            "provided policy documents. Please check the policy document "
            "directly or contact your insurer for clarification."
        )
    else:
        decision = "answer"
        safe_message = None
    
    result = {
        "should_answer": decision == "answer",
        "risk": risk,
        "confidence": confidence,
        "final_decision": decision,
        "safe_message": safe_message
    }
    
    logger.info(f"Query evaluation: decision={decision}")
    return result