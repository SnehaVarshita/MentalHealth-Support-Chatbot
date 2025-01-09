import streamlit as st
from auth import signup_user, login_user
from bot import pred_class, get_response, bert_model, model, label_encoder, responses_dict

# Define Chatbot interaction functions
def chat_with_bot(message):
    intent = pred_class(message, bert_model, model, label_encoder)
    response = get_response(intent, responses_dict)
    return response

# URLs for custom icons
CHATGPT_ICON_URL = "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"  # Chatbot icon
USER_ICON_URL = "https://cdn-icons-png.flaticon.com/512/709/709699.png"       # User icon

# State management for session
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "message_input" not in st.session_state:
    st.session_state.message_input = ""
if "show_history" not in st.session_state:
    st.session_state.show_history = False
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {"email": "", "dob": ""}

# Login and Signup Page
if not st.session_state.authenticated:
    st.title("ChatBot Login")

    if "show_signup" not in st.session_state:
        st.session_state.show_signup = False

    # Login Form
    if not st.session_state.show_signup:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            success, message = login_user(username, password)
            if success:
                st.session_state.authenticated = True
                st.success("Logged in successfully.")
            else:
                st.error(message)

        # Signup link
        if st.button("Create an Account"):
            st.session_state.show_signup = True

    # Signup Form
    else:
        st.subheader("Signup")
        new_username = st.text_input("Create Username")
        new_password = st.text_input("Create Password", type="password")
        if st.button("Signup"):
            success, message = signup_user(new_username, new_password)
            if success:
                st.session_state.show_signup = False
                st.success("Account created successfully. Please log in.")
            else:
                st.error(message)
        if st.button("Back to Login"):
            st.session_state.show_signup = False

# Chat Interface
else:
    # Sidebar for navigation and options
    with st.sidebar:
        st.markdown("## Navigation")
        chat_selected = st.radio("Navigation", options=["Chat", "History", "Profile"], index=0)

        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.chat_history = []
            st.success("Logged out successfully.")
            st.rerun()

    # Header
    st.title("Mental Health Support Chatbot")
    #st.caption("Type '0' to end the conversation.")

    # Chat Section
    if chat_selected == "Chat":
        # Display chat history with custom icons
        for message, is_user in reversed(st.session_state.chat_history):
            if is_user:
                st.image(USER_ICON_URL, width=30)  # Reduce icon size
                st.markdown(f"<div style='display: inline-block; padding-left: 10px; font-weight: bold;'>{message}</div>", unsafe_allow_html=True)
            else:
                st.image(CHATGPT_ICON_URL, width=30)  # Reduce icon size
                st.markdown(f"<div style='display: inline-block; padding-left: 10px; font-weight: bold;'>{message}</div>", unsafe_allow_html=True)

        # Input for new messages
        user_message = st.text_input("Type your message here:", value=st.session_state.message_input)

        # Send button logic
        if st.button("Send"):
            if user_message:
                # Append user's message to chat history
                st.session_state.chat_history.append((user_message, True))

                # Get bot's response and add to chat history
                bot_response = chat_with_bot(user_message)
                st.session_state.chat_history.append((bot_response, False))

                # Clear the input field
                st.session_state.message_input = ""

                # Trigger a rerun to refresh the interface
                st.rerun()

    # History Section
    elif chat_selected == "History":
        st.subheader("Conversation History")
        
        if not st.session_state.chat_history:
            st.info("No conversation history found.")
        else:
            # Display the entire chat history with icons
            for message, is_user in st.session_state.chat_history:
                if is_user:
                    st.image(USER_ICON_URL, width=30)  # Reduce icon size
                    st.markdown(f"<div style='display: inline-block; padding-left: 10px; font-weight: bold;'>{message}</div>", unsafe_allow_html=True)
                else:
                    st.image(CHATGPT_ICON_URL, width=30)  # Reduce icon size
                    st.markdown(f"<div style='display: inline-block; padding-left: 10px; font-weight: bold;'>{message}</div>", unsafe_allow_html=True)

    # Profile Section
    elif chat_selected == "Profile":
        st.subheader("User Profile")
        
        # Display and edit user profile information
        email = st.text_input("Email", value=st.session_state.user_profile.get("email", ""))
        dob = st.text_input("Date of Birth (YYYY-MM-DD)", value=st.session_state.user_profile.get("dob", ""))

        if st.button("Save Profile"):
            st.session_state.user_profile["email"] = email
            st.session_state.user_profile["dob"] = dob
            st.success("Profile updated successfully!")

        # Display updated profile information
        st.write(f"**Email**: {st.session_state.user_profile['email']}")
        st.write(f"**Date of Birth**: {st.session_state.user_profile['dob']}")

# User authentication functions
def signup_user(username, password):
    try:
        with open("users.txt", "a") as file:
            file.write(f"{username},{password}\n")
        return True, "Account created successfully!"
    except Exception as e:
        return False, str(e)

def login_user(username, password):
    try:
        with open("users.txt", "r") as file:
            users = file.readlines()
            for user in users:
                stored_username, stored_password = user.strip().split(",")
                if stored_username == username and stored_password == password:
                    return True, "Login successful!"
        return False, "Invalid credentials"
    except Exception as e:
        return False, str(e)

