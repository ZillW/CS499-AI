import os
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os


os.environ["OPENAI_API_KEY"] = "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"


# --- Utility Functions ---

def validate_input(user_input, valid_options=None):
    """
    Validates the given input based on an optional list of valid options.
    If valid_options is provided, it returns the matched option in lowercase.
    Otherwise returns the input (if non-empty).
    """
    if valid_options:
        for option in valid_options:
            if option.lower() in user_input.lower():
                return option.lower()
        return None
    else:
        return user_input.strip() if user_input.strip() else None


def add_message(role, message):
    """
    Appends a new message to the session state's messages list.
    role: "assistant" or "user"
    """
    st.session_state.messages.append((role, message))


def display_chat_history():
    """
    Displays the conversation messages in a chat-like format.
    """
    for role, message in st.session_state.messages:
        if role == "assistant":
            st.markdown(f"**AdMaven:** {message}")
        else:
            st.markdown(f"**You:** {message}")


# --- Conversation Processing ---

def process_step(user_input):
    """
    Process the conversation step based on the current state.
    Each step validates and stores input, asks next question,
    and calls LLM chains when needed.
    """
    # Step 0: Ask for ad type ("wanted" or "have")
    if st.session_state.step == 0:
        validated = validate_input(user_input, valid_options=["wanted", "have"])
        if validated is None:
            add_message("assistant",
                        "Sorry, I didn't understand that. Please respond with one of the following options: wanted, have")
        else:
            st.session_state.data["ad_type"] = validated
            st.session_state.step = 1
            # Step 1: Ask for category
            if validated == "wanted":
                add_message("assistant", "Are you looking for a person, a product, or a service?")
            else:
                add_message("assistant", "Are you advertising a person, a product, or a service?")

    # Step 1: Ask for category
    elif st.session_state.step == 1:
        validated = validate_input(user_input, valid_options=["person", "product", "service"])
        if validated is None:
            add_message("assistant",
                        "Sorry, I didn't understand that. Please respond with one of the following options: person, product, service")
        else:
            st.session_state.data["category"] = validated
            st.session_state.step = 2
            add_message("assistant",
                        "Please provide the details of what you are advertising, please provide as much detail as possible.")

    # Step 2: Ask for basic details
    elif st.session_state.step == 2:
        st.session_state.data["details"] = user_input
        st.session_state.step = 3
        if st.session_state.data["ad_type"] == "wanted":
            add_message("assistant", "What is your timeline for receiving valid leads? (e.g., within 30 days)")
        else:
            add_message("assistant", "What is the availability timeline for your offer? (e.g., available from Q2)")

    # Step 3: Ask for timeline
    elif st.session_state.step == 3:
        st.session_state.data["timeline"] = user_input
        st.session_state.step = 4
        if st.session_state.data["ad_type"] == "wanted":
            add_message("assistant", "What bounty (in credits) are you offering for a valid introduction?")
        else:
            add_message("assistant", "At what price (in credits) are you offering this contact/lead?")

    # Step 4: Ask for bounty (wanted) or price (have)
    elif st.session_state.step == 4:
        if st.session_state.data["ad_type"] == "wanted":
            st.session_state.data["bounty"] = user_input
        else:
            st.session_state.data["price"] = user_input
        st.session_state.step = 5

        # --- Generate Clarifying Questions ---
        ad_template = """Ad Format Template
Title:
Short, clear headline describing the main focus
(e.g., “Wanted: Expert Data Analyst for Tech Startup” or “Have: Direct Contact to CFO in Manufacturing”).

Brief Summary (1–2 sentences):
High-level overview of what you need or offer.
Example (Wanted): “Seeking a decision-maker in HR for mid-sized logistics companies with immediate hiring needs.”
Example (Have): “I can introduce you to a marketing VP looking to outsource SEO services next quarter.”

Specific Details:
- Scope / Requirements (for Wanted ads) or Offer Highlights (for Have ads)
- Location (regional, global, or remote)
- Industry/Domain (if any)
- Key Credentials (title, experience, product specs, etc.)

Timeline:
State any deadlines or time constraints.
Example (Wanted): “Need introductions within 30 days—planning marketing outreach next month.”
Example (Have): “Availability starting Q2; early leads best accepted by March 15.”

Good-to-Haves / Additional Requirements:
Any non-mandatory yet valuable attributes.
Example (Wanted): “Offering 100 credits for a valid introduction.”
Example (Have): “Priced at 75 credits for direct contact details and warm handoff.”

Contact & Validation Info:
Short note on how funds will be released or validation performed.
Example: “Escrow funds are held until I confirm the lead is accurate” or “Payment released upon confirmation that the introduction worked.”

Closing Statement:
Prompt responders to take action.
Example (Wanted): “If you can provide these leads, please respond with your confidence level and how you know them!”
Example (Have): “Message me if you’re interested—I’ll share further specifics and finalize escrow details.”
"""
        clarifying_prompt_text = f""" You are an expert assistant for creating a custom ad.
The ad will follow the template provided below:
{ad_template}
This is the information we know so far (provided below):
Ad Type: {st.session_state.data["ad_type"].title()}
Category: {st.session_state.data["category"].title()}
Details: {st.session_state.data["details"]}
Timeline: {st.session_state.data["timeline"]}
{'Bounty: ' + st.session_state.data["bounty"] + " credits" if st.session_state.data["ad_type"] == 'wanted' else "Price: " + st.session_state.data["price"] + " credits"}
Based on the above information, list 1-5 clarifying, probing questions that you would ask to gather any missing or vital details in order to create a complete and precise advertisement based on the provided template. If the provided ad type is Wanted, your questions will be asked to a user who is looking for something. If the provided ad type is Have, your questions will be asked to a user who is already has something which they are advertising. You must include questions about location, budget, and payment if not already provided. Do not include any summaries of the provided details, and do not provide extra commentary. Only output the list of questions.
"""
        clarifying_prompt = ChatPromptTemplate.from_template(clarifying_prompt_text)
        clarifying_chain = clarifying_prompt | st.session_state.model
        generated_clarifications = clarifying_chain.invoke({"question": ""})
        st.session_state.data["clarifying_questions"] = generated_clarifications.strip()
        add_message("assistant",
                    "I have some follow-up questions for you:\n" + st.session_state.data["clarifying_questions"])
        st.session_state.step = 6
        add_message("assistant",
                    "Please provide answers to the above questions (or type 'No additional details required').")

    # Step 5: Process clarifying questions answer
    elif st.session_state.step == 6:
        st.session_state.data["clarifying_answers"] = user_input
        st.session_state.step = 7

        # --- Generate Final Ad ---
        ad_template = """Ad Format Template
Title:
Short, clear headline describing the main focus
(e.g., “Wanted: Expert Data Analyst for Tech Startup” or “Have: Direct Contact to CFO in Manufacturing”).

Brief Summary (1–2 sentences):
High-level overview of what you need or offer.
Example (Wanted): “Seeking a decision-maker in HR for mid-sized logistics companies with immediate hiring needs.”
Example (Have): “I can introduce you to a marketing VP looking to outsource SEO services next quarter.”

Specific Details:
- Scope / Requirements (for Wanted ads) or Offer Highlights (for Have ads)
- Location (regional, global, or remote)
- Industry/Domain (if any)
- Key Credentials (title, experience, product specs, etc.)

Timeline:
State any deadlines or time constraints.
Example (Wanted): “Need introductions within 30 days—planning marketing outreach next month.”
Example (Have): “Availability starting Q2; early leads best accepted by March 15.”

Good-to-Haves / Additional Requirements:
Any non-mandatory yet valuable attributes.
Example (Wanted): “Offering 100 credits for a valid introduction.”
Example (Have): “Priced at 75 credits for direct contact details and warm handoff.”

Contact & Validation Info:
Short note on how funds will be released or validation performed.
Example: “Escrow funds are held until I confirm the lead is accurate” or “Payment released upon confirmation that the introduction worked.”

Closing Statement:
Prompt responders to take action.
Example (Wanted): “If you can provide these leads, please respond with your confidence level and how you know them!”
Example (Have): “Message me if you’re interested—I’ll share further specifics and finalize escrow details.”
"""
        final_prompt_template = f"""
Below is the information you provided:
Ad Type: {st.session_state.data["ad_type"].title()}
Category: {st.session_state.data["category"].title()}
Details: {st.session_state.data["details"]}
Timeline: {st.session_state.data["timeline"]}
{'Bounty: ' + st.session_state.data["bounty"] + " credits" if st.session_state.data["ad_type"] == 'wanted' else "Price: " + st.session_state.data["price"] + " credits"}
Additional Clarifications: {st.session_state.data["clarifying_answers"]}

Using the ad format template below, generate the final ad text, you must include all of the below sections:

{ad_template}
"""
        final_prompt = ChatPromptTemplate.from_template(final_prompt_template)
        final_chain = final_prompt | st.session_state.model
        final_ad = final_chain.invoke(
            {"question": "Please generate the final ad text using the details provided above."})
        st.session_state.data["final_ad"] = final_ad
        add_message("assistant", "\n----- FINAL AD -----\n" + final_ad)
        # Conversation completed.
        st.session_state.step = 7

    # If conversation is complete, no further steps.
    else:
        add_message("assistant", "Thank you! The conversation is complete.")


# --- Main App ---

def main():
    st.title("AdMaven Chatbot")

    # Initialize session state variables if not already initialized.
    if "model" not in st.session_state:
        st.session_state.model = OpenAI()
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.step = 0  # Conversation step tracker.
        st.session_state.data = {}  # To hold user's answers.
        # Initial greeting and question.
        add_message("assistant",
                    "Hello! I am AdMaven, your expert assistant for creating a custom ad.\n"
                    "I will help you gather all the necessary details to craft a precise and effective advertisement.\n"
                    "Are you posting a wanted ad (need something) or a have ad (have something to advertise)?"
                    )

    # Display current conversation history.
    display_chat_history()

    # Chat input: using streamlit chat_input if available, otherwise fallback to text_input.
    if hasattr(st, "chat_input"):
        user_input = st.chat_input("Your message")
    else:
        user_input = st.text_input("Your message")

    if user_input:
        add_message("user", user_input)
        process_step(user_input)
        st.rerun()  # Rerun to update the conversation display.


if __name__ == "__main__":
    main()


# ========== Flask ==========
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/generate_ad', methods=['POST'])
def generate_ad():
    data = request.get_json()
    answers = data.get("answers", [])

    final_prompt = (
        f"Wanted Ad Summary:\n"
        f"Ad Type: {answers[0]}\n"
        f"Category: {answers[1]}\n"
        f"Goal: {answers[2]}\n"
        f"Timeline: {answers[3]}\n"
        f"Location & Industry: {answers[4]}\n"
        f"Bounty: {answers[5]}\n"
        f"Lead Details: {answers[6]}\n"
        f"Additional Requirements: {answers[7]}\n"
    )

    from langchain_openai import OpenAI
    from langchain.prompts import ChatPromptTemplate
    llm = OpenAI()
    prompt_template = ChatPromptTemplate.from_template("{summary}")
    chain = prompt_template | llm
    result = chain.invoke({"summary": final_prompt})

    return jsonify({"ad_output": result})

if __name__ == '__main__':
    import sys
    if "runserver" in sys.argv:
        app.run(host="0.0.0.0", port=5000)
    else:
        import streamlit as st
