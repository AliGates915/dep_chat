# main.py

# =========================================================
# STEP 1: Imports & Init
# =========================================================
import os
import difflib
import threading
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from openai import OpenAI

load_dotenv()

# OpenAI & API config
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CAT_URL = os.getenv("CATEGORY_API_URL")
ITEM_URL = os.getenv("ITEMS_API_BASE")
BILL_API_URL = os.getenv("BILL_API_URL")

# =========================================================
# STEP 2: FastAPI Init & Middleware
# =========================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can change this to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================================================
# STEP 3: Session Store (In-memory for simplicity)
# =========================================================
sessions = {}  # session_id: dict with user_info, cart, etc.

def reset_session(session_id):
    time.sleep(30)
    sessions.pop(session_id, None)

# =========================================================
# STEP 4: Utility Functions
# =========================================================

def get_categories():
    try:
        res = requests.get(CAT_URL)
        return res.json() if res.ok else []
    except:
        return []

def smart_match_category(user_input, categories):
    user_input = user_input.strip().lower()
    category_names = [c["categoryName"] for c in categories]

    for c in categories:
        if c["categoryName"].lower() == user_input:
            return c

    closest = difflib.get_close_matches(user_input, [c.lower() for c in category_names], n=1, cutoff=0.6)
    if closest:
        for c in categories:
            if c["categoryName"].lower() == closest[0]:
                return c
    return None

def get_ai_response(user_msg):
    context = """
You are a shopping assistant for AITOPIA Shopping Mart. 
You only help with the following:
- Categories: Electronics, Vegetables & Fruits, Clothes
- Payment & Delivery: Cash on Delivery, Nationwide delivery
- About: AITOPIA is an online mart.

If user asks anything else, politely say you only assist with shopping-related queries.
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": user_msg}
        ]
    )
    return res.choices[0].message.content

# =========================================================
# STEP 5: API Models
# =========================================================
class ChatRequest(BaseModel):
    session_id: str
    message: str

class BillRequest(BaseModel):
    session_id: str

# =========================================================
# STEP 6: POST /chat — Main Chat Logic
# =========================================================
@app.post("/chat")
async def chat(req: ChatRequest):
    msg = req.message.strip()
    sid = req.session_id

    if sid not in sessions:
        sessions[sid] = {
            "login_step": 0,
            "user_info": {},
            "cart": {},
            "items": [],
            "categories": get_categories(),
            "selected_cat": None,
        }

    state = sessions[sid]
    reply = ""
    logout = False

    # -------------------------------
    # Login Flow
    # -------------------------------
    if state["login_step"] == 0:
        if msg.replace(" ", "").isalpha():
            state["user_info"]["name"] = msg
            state["login_step"] = 1
            reply = "📞 Please enter your phone number (starting with 03):"
        else:
            reply = "❌ Name should only contain letters. Try again."

    elif state["login_step"] == 1:
        digits = ''.join(filter(str.isdigit, msg))
        if (len(digits) in [10, 11]) and digits.startswith("03"):
            state["user_info"]["phone"] = digits
            state["login_step"] = 2
            reply = "🏠 Please enter your address:"
        else:
            reply = "❌ Phone must start with 03 and be 10–11 digits."

    elif state["login_step"] == 2:
        if msg:
            state["user_info"]["address"] = msg
            state["login_step"] = 3
            cat_names = [c["categoryName"] for c in state["categories"] if c.get("isEnable")]
            reply = f"✅ You're logged in!\nAvailable categories: {', '.join(cat_names)}\n\nType a category name to view items."
        else:
            reply = "❌ Address cannot be empty."

    # -------------------------------
    # After Login: Chat Logic
    # -------------------------------
    else:
        msg_lower = msg.lower()

        # Show bill
        if msg_lower in ["bill", "show bill"]:
            cart = state["cart"]
            if not cart:
                reply = "🛒 Your cart is empty."
            else:
                lines = []
                total = 0
                for name, info in cart.items():
                    qty = info["quantity"]
                    price = info["price"]
                    subtotal = qty * price
                    lines.append(f"- {name}: {qty} x {price} = {subtotal} Rs")
                    total += subtotal
                reply = "🧾 Your Cart:\n" + "\n".join(lines) + f"\n\n💰 Total: {total} Rs"

        # Add item
        elif msg_lower.startswith("add "):
            parts = msg.split()
            qty = 1
            item_name = ""

            if len(parts) >= 2:
                if parts[1].isdigit():
                    qty = int(parts[1])
                    item_name = " ".join(parts[2:])
                else:
                    item_name = " ".join(parts[1:])

            matched = None
            for it in state["items"]:
                if it["itemName"].lower() == item_name.lower():
                    matched = it
                    break

            if matched:
                if matched["itemName"] in state["cart"]:
                    state["cart"][matched["itemName"]]["quantity"] += qty
                else:
                    state["cart"][matched["itemName"]] = {
                        "price": matched.get("price", 0),
                        "quantity": qty
                    }
                reply = f"✅ Added {qty} x {matched['itemName']} to your cart."
            else:
                reply = f"❌ Item '{item_name}' not found in current category."

        # Checkout
        elif msg_lower == "checkout":
            cart = state["cart"]
            if not cart:
                reply = "🛒 Your cart is empty."
            else:
                total = sum(info["price"] * info["quantity"] for info in cart.values())
                lines = [f"{k}: {v['quantity']} x {v['price']} = {v['quantity']*v['price']} Rs" for k, v in cart.items()]
                bill = "\n".join(lines)
                user = state["user_info"]
                reply = f"🧾 Final Bill:\n{bill}\n\n💰 Total: {total} Rs\n👤 {user['name']}\n📞 {user['phone']}\n🏠 {user['address']}"
                logout = True

        # Category selection
        else:
            matched_cat = smart_match_category(msg, state["categories"])
            if matched_cat:
                try:
                    items = requests.get(f"{ITEM_URL}/{matched_cat['categoryName']}").json()
                    if isinstance(items, list):
                        state["selected_cat"] = matched_cat
                        state["items"] = items
                        item_list = "\n".join([f"- {i['itemName']} ({i.get('price', 'N/A')} Rs)" for i in items])
                        reply = f"📦 Items in {matched_cat['categoryName']}:\n{item_list}\nType `add <qty> <item>` to add to cart."
                    else:
                        reply = f"⚠️ No items found for {matched_cat['categoryName']}."
                except:
                    reply = "⚠️ Failed to fetch items."
            else:
                reply = get_ai_response(msg)

    # Auto logout
    if logout:
        try:
            requests.post(BILL_API_URL, json={
                "user": state["user_info"],
                "cart": state["cart"]
            })
        except:
            pass
        threading.Thread(target=reset_session, args=(sid,), daemon=True).start()

    return {"reply": reply, "session": state}

# =========================================================
# STEP 7: Serve index.html
# =========================================================
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")
