# main.py
# Changes summary:
# - ADDED: parse_payment_method helper to normalize user payment replies.
# - CHANGED: fn_checkout now checks for missing payment method and includes "paymentMethod" in returned JSON and POST payload.
# - ADDED: session fields "payment_method" and "awaiting_payment_method" on session creation.
# - CHANGED: call_openai_and_maybe_execute now returns "function_result" so /chat can inspect function outputs (e.g., needs_payment_method).
# - CHANGED: /chat now intercepts awaiting_payment_method replies, normalizes & saves the payment method, and finalizes checkout accordingly.
# - CHANGED: fallback and function-call checkout paths updated to request payment method (set awaiting flag) instead of finalizing immediately when missing.

# =========================================================
# STEP 1: Imports & Init
# =========================================================
import os
import re
import json
import difflib
import threading
import time
import traceback
from typing import Any, Dict, Optional  # ADDED: Optional already in original import list

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

# OpenAI & API config
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CAT_URL = os.getenv("CATEGORY_API_URL")
ITEM_URL = os.getenv("ITEMS_API_BASE")
BILL_API_URL = os.getenv("BILL_API_URL")

MODEL_NAME = "gpt-4o-mini"  # kept same as original usage

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
# - Keep your original session structure and auto logout
# =========================================================
sessions = {}  # session_id: dict with user_info, cart, etc.

def reset_session(session_id):
    """Auto-logout helper (same as original): waits 30s then removes session."""
    time.sleep(30)
    sessions.pop(session_id, None)

# =========================================================
# STEP 4: Utility Functions (keep originals + new payment parser)
# =========================================================
def get_categories():
    """Calls CATEGORY_API_URL and returns JSON list or empty list on failure."""
    try:
        res = requests.get(CAT_URL)
        return res.json() if res.ok else []
    except:
        return []

def smart_match_category(user_input, categories):
    """
    Existing fuzzy match helper for category selection.
    Returns full category dict if found, else None.
    """
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

# ADDED: helper to parse user's payment method replies
def parse_payment_method(msg: str) -> Optional[str]:
    """
    ADDED: Normalize various user replies to one of:
      - "Cash on Delivery"
      - "Online Transfer"
    Returns normalized string or None if not matched.
    """
    if not msg:
        return None
    s = msg.strip().lower()
    # cash patterns
    if any(k in s for k in ["cash on delivery", "cash on", "cash", "cod", "c.o.d"]):
        return "Cash on Delivery"
    # online/transfer patterns (bank, card, easypaisa, jazzcash etc)
    if any(k in s for k in ["online", "transfer", "bank", "card", "visa", "mastercard", "easypaisa", "jazzcash", "mobile", "payment"]):
        return "Online Transfer"
    # numeric choices fallback
    if s in ["1", "one"]:
        return "Cash on Delivery"
    if s in ["2", "two"]:
        return "Online Transfer"
    return None

# =========================================================
# STEP 5: API Models
# =========================================================
class ChatRequest(BaseModel):
    session_id: str
    message: str

class BillRequest(BaseModel):
    session_id: str

# =========================================================
# STEP 6: OpenAI function definitions & helper execution mapping
# - Define the functions schema that will be passed to OpenAI
# - Implement local handlers that perform the actual actions on `state`
# =========================================================

# Function definitions (JSON-schema) sent to OpenAI so it can decide to call them.
OPENAI_FUNCTIONS = [
    {
        "name": "get_categories",
        "description": "Return list of available categories (enabled or returned by the CATEGORY API). No arguments.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_items",
        "description": "Fetch items in a category. Args: categoryName (string).",
        "parameters": {
            "type": "object",
            "properties": {"categoryName": {"type": "string"}},
            "required": ["categoryName"],
        },
    },
    {
        "name": "add_to_cart",
        "description": "Add an item to the cart. Args: itemName (string), quantity (integer, default 1).",
        "parameters": {
            "type": "object",
            "properties": {
                "itemName": {"type": "string"},
                "quantity": {"type": "integer", "minimum": 1},
            },
            "required": ["itemName"],
        },
    },
    {
        "name": "show_bill",
        "description": "Return a summary of the current cart and total amount. No arguments.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "checkout",
        "description": "Finalize the bill (send to BILL_API_URL) and prepare for logout. No arguments.",
        "parameters": {"type": "object", "properties": {}},
    },
    # optional: remove_from_cart (not part of original code but harmless and useful)
    {
        "name": "remove_from_cart",
        "description": "Remove an item from the cart or reduce its quantity. Args: itemName (string), quantity (integer, default 1).",
        "parameters": {
            "type": "object",
            "properties": {
                "itemName": {"type": "string"},
                "quantity": {"type": "integer", "minimum": 1},
            },
            "required": ["itemName"],
        },
    },
]

# Local function implementations (these mutate `state` as your original code expects)
def fn_get_categories(state: Dict[str, Any]) -> Dict[str, Any]:
    cats = get_categories()
    # store categories in session state to keep them consistent
    state["categories"] = cats
    cat_names = [c.get("categoryName", "") for c in cats if c.get("isEnable", True)]
    return {"categories": cat_names, "raw": cats}

def fn_get_items(state: Dict[str, Any], categoryName: str) -> Dict[str, Any]:
    try:
        items = requests.get(f"{ITEM_URL}/{categoryName}").json()
    except Exception:
        items = []
    if isinstance(items, list):
        state["selected_cat"] = {"categoryName": categoryName}
        state["items"] = items
        items_list = [{"itemName": i.get("itemName"), "price": i.get("price", "N/A")} for i in items]
        return {"items": items_list}
    else:
        return {"items": [], "warning": f"No items found for {categoryName}"}

def _match_item_in_state(state: Dict[str, Any], item_name: str):
    """Improved fuzzy matcher for items (handles plurals and partial matches)."""
    if not state.get("items"):
        return None

    target = item_name.strip().lower()

    # exact match
    for it in state["items"]:
        if it.get("itemName", "").lower() == target:
            return it

    # singular/plural match
    for it in state["items"]:
        name = it.get("itemName", "").lower()
        if name.rstrip('s') == target.rstrip('s'):
            return it

    # partial match
    for it in state["items"]:
        if target in it.get("itemName", "").lower():
            return it

    # fuzzy match (difflib)
    names = [it.get("itemName", "") for it in state["items"]]
    closest = difflib.get_close_matches(target, [n.lower() for n in names], n=1, cutoff=0.5)
    if closest:
        for it in state["items"]:
            if it.get("itemName", "").lower() == closest[0]:
                return it

    return None

def fn_add_to_cart(state: Dict[str, Any], itemName: str, quantity: int = 1) -> Dict[str, Any]:
    """
    Matches item in current state['items'] and updates state['cart'].
    Behaves like your original 'add ' command logic.
    """
    matched = _match_item_in_state(state, itemName)
    if not matched:
        return {"success": False, "message": f"Item '{itemName}' not found in current category."}
    name = matched["itemName"]
    price = matched.get("price", 0)
    if name in state["cart"]:
        state["cart"][name]["quantity"] += quantity
    else:
        state["cart"][name] = {"price": price, "quantity": quantity}
    return {"success": True, "message": f"Added {quantity} x {name} to your cart."}

def fn_remove_from_cart(state: Dict[str, Any], itemName: str, quantity: int = 1) -> Dict[str, Any]:
    """Optional: remove or reduce quantity from cart."""
    cart = state["cart"]
    # find exact/fuzzy match in cart keys
    keys = list(cart.keys())
    match = None
    for k in keys:
        if k.lower() == itemName.lower():
            match = k
            break
    if not match:
        closest = difflib.get_close_matches(itemName.lower(), [k.lower() for k in keys], n=1, cutoff=0.6)
        if closest:
            for k in keys:
                if k.lower() == closest[0]:
                    match = k
                    break
    if not match:
        return {"success": False, "message": f"Item '{itemName}' not present in cart."}
    if cart[match]["quantity"] <= quantity:
        del cart[match]
        return {"success": True, "message": f"Removed {match} from cart."}
    else:
        cart[match]["quantity"] -= quantity
        return {"success": True, "message": f"Reduced {match} quantity by {quantity}."}

def fn_show_bill(state: Dict[str, Any]) -> Dict[str, Any]:
    cart = state.get("cart", {})
    if not cart:
        return {"empty": True, "message": "Your cart is empty."}
    lines = []
    total = 0
    for name, info in cart.items():
        qty = info["quantity"]
        price = info["price"]
        subtotal = qty * price
        lines.append({"name": name, "quantity": qty, "price": price, "subtotal": subtotal})
        total += subtotal
    return {"empty": False, "lines": lines, "total": total}

def fn_checkout(state: Dict[str, Any], sid: str) -> Dict[str, Any]:
    """
    CHANGED: If payment method not present in session state, do NOT finalize checkout.
             Instead return needs_payment_method=True and a prompt message.
    CHANGED: When finalizing, include 'paymentMethod' in posted JSON and returned dict.
    """
    cart = state.get("cart", {})
    if not cart:
        return {"success": False, "message": "Your cart is empty."}

    # CHANGED: require a payment method before finalizing
    payment_method = state.get("payment_method")
    if not payment_method:
        return {
            "success": False,
            "needs_payment_method": True,
            "message": "Do you want to pay via Cash on Delivery or Online Transfer?"
        }

    total = sum(info["price"] * info["quantity"] for info in cart.values())
    lines = [{"name": k, "quantity": v["quantity"], "price": v["price"], "subtotal": v["quantity"]*v["price"]} for k, v in cart.items()]
    user = state.get("user_info", {})

    # CHANGED: Try to POST bill to BILL_API_URL (best-effort) and include paymentMethod
    try:
        requests.post(BILL_API_URL, json={"user": user, "cart": cart, "paymentMethod": payment_method})
    except Exception:
        # we intentionally swallow errors (same as original)
        pass
    # schedule logout (same as original)
    threading.Thread(target=reset_session, args=(sid,), daemon=True).start()
    # return a friendly summary including payment method
    return {
        "success": True,
        "message": "Checkout complete. Your order has been placed.",
        "bill": {"lines": lines, "total": total},
        "user": user,
        "paymentMethod": payment_method,  # CHANGED: include normalized payment method
    }

# Mapping function name -> implementing callable
FUNCTION_IMPLEMENTATIONS = {
    "get_categories": lambda state, sid=None, args={}: fn_get_categories(state),
    "get_items": lambda state, sid=None, args={}: fn_get_items(state, args.get("categoryName", "")),
    "add_to_cart": lambda state, sid=None, args={}: fn_add_to_cart(state, args.get("itemName", ""), int(args.get("quantity", 1))),
    "show_bill": lambda state, sid=None, args={}: fn_show_bill(state),
    "checkout": lambda state, sid=None, args={}: fn_checkout(state, sid),
    "remove_from_cart": lambda state, sid=None, args={}: fn_remove_from_cart(state, args.get("itemName", ""), int(args.get("quantity", 1))),

}

# =========================================================
# STEP 7: OpenAI conversation helper (function-calling flow)
# - This helper sends the message to OpenAI, includes the function schema,
#   processes any function_call returned by OpenAI, executes the local function,
#   and then sends the function result back to OpenAI to get a final reply.
# - If no function_call is produced by OpenAI, we return the assistant content (if any)
#   and the 'function_call' as None.
# =========================================================
def call_openai_and_maybe_execute(user_msg: str, state: Dict[str, Any], sid: str) -> Dict[str, Any]:
    """
    Handles user input by calling OpenAI and optionally executing backend functions.

    Returns a dict:
      {
        "assistant_content": str or None,
        "function_call": {"name": str, "arguments": dict} or None,
        "final_content": str or None,  # present if we performed function->finalize cycle
        "raw": <raw openai response object or dict for debugging>,
        "function_result": <the result from local function execution, if any>  # CHANGED: added
      }
    """

    # -------------------------------
    # Step 1: Load categories so AI knows what exists
    # -------------------------------
    cat_names = [c.get("categoryName", "") for c in (state.get("categories") or [])]
    if not cat_names:
        try:
            cat_list = get_categories()
            state["categories"] = cat_list
            cat_names = [c.get("categoryName", "") for c in cat_list]
        except:
            cat_names = []

    # -------------------------------
    # Step 2: Build system context (system prompt)
    # -------------------------------
    # 🔹 This has been updated for:
    #    - Bilingual behavior
    #    - Login flow (AI validates name, backend only saves after confirmation)
    #    - Service explanation (instead of rejecting)
    system_context = (
        "You are a bilingual shopping assistant for Point of Sale (POS).\n"
        "Your job is to help customers with shopping-related queries, including:\n"
        "- Browse categories\n"
        "- View items with prices\n"
        "- Add or remove items from cart\n"
        "- Show bill\n"
        "- Checkout\n\n"

        "NATURAL LANGUAGE RULES:\n"
        "- Understand variations like: 'I want to buy vegetables' → treat as category 'vegetables'.\n"
        "- 'Add 2 carrots', 'give me 2 carrots', 'please add 2 carrots' → all mean add_to_cart(item='carrots', qty=2).\n"
        "- Do not repeat questions that the user already answered.\n\n"

        "SMALL TALK RULES:\n"
        "- If the user greets (hi/hello/salam), reply politely but also guide them back to shopping.\n"
        "- If the user asks for services, explain: You can browse categories, view items with prices, add to cart, see bill, and checkout.\n"
        "- Only reply with 'Sorry, I can only help you with shopping at Point of Sale' if the request is truly unrelated (e.g. jokes, trivia, weather).\n\n"

        "BILINGUAL RULE:\n"
        "- If user speaks in Roman Urdu, reply in Roman Urdu.\n"
        "- If user speaks in English, reply in English.\n"
    )

    # -------------------------------
    # Step 3: Build payload for AI
    # -------------------------------
    user_payload = (
        f"Session login_step={state.get('login_step', 0)}. "
        f"Available categories: {', '.join(cat_names)}. "
        f"Raw message: {user_msg}"
    )

    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": user_payload},
    ]

    # -------------------------------
    # Step 4: Call OpenAI
    # -------------------------------
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            functions=OPENAI_FUNCTIONS,
            function_call="auto",
            temperature=0.2,
        )
    except Exception as e:
        return {
            "assistant_content": None,
            "function_call": None,
            "final_content": None,
            "raw": {"error": str(e)},
            "function_result": None,  # CHANGED: include for consistency
        }

    # -------------------------------
    # Step 5: Normalize AI response
    # -------------------------------
    raw_choice = None
    try:
        raw_choice = response.choices[0].message
    except Exception:
        raw_choice = response

    func_call = None
    assistant_content = None
    try:
        if isinstance(raw_choice, dict) and raw_choice.get("function_call"):
            func_call = raw_choice["function_call"]
        else:
            if hasattr(raw_choice, "get") and raw_choice.get("function_call"):
                func_call = raw_choice.get("function_call")
            elif getattr(raw_choice, "function_call", None):
                func_call = raw_choice.function_call
            else:
                assistant_content = (
                    raw_choice.get("content")
                    if isinstance(raw_choice, dict)
                    else getattr(raw_choice, "content", None)
                )
    except Exception:
        try:
            func_call = raw_choice["function_call"]
        except Exception:
            assistant_content = (
                raw_choice.get("content") if isinstance(raw_choice, dict) else None
            )

    # -------------------------------
    # Step 6: Handle function calls
    # -------------------------------
    if func_call:
        fname = (
            func_call.get("name")
            if isinstance(func_call, dict)
            else getattr(func_call, "name", None)
        )
        raw_args = (
            func_call.get("arguments")
            if isinstance(func_call, dict)
            else getattr(func_call, "arguments", "{}")
        )

        try:
            if isinstance(raw_args, str):
                args = json.loads(raw_args) if raw_args.strip() else {}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
        except Exception:
            args = {}

        # ⚠️ IMPORTANT FIX:
        # If login_step < 3 (user not logged in yet), do NOT execute shopping functions.
        if state.get("login_step", 0) < 3:
            return {
                "assistant_content": assistant_content,
                "function_call": {"name": fname, "arguments": args},
                "final_content": None,
                "raw": response,
                "function_result": None,  # CHANGED: include for consistency
            }

        # Otherwise, execute backend function
        impl = FUNCTION_IMPLEMENTATIONS.get(fname)
        if not impl:
            function_result = {
                "success": False,
                "error": f"No local implementation for function '{fname}'.",
            }
        else:
            try:
                function_result = (
                    impl(state, sid=sid, args=args)
                    if callable(impl)
                    else {"error": "Invalid implementation"}
                )
            except Exception as e:
                function_result = {
                    "success": False,
                    "error": f"Exception during function execution: {str(e)}",
                    "trace": traceback.format_exc(),
                }

        # Send function result back to AI for natural response
        followup_messages = messages.copy()
        followup_messages.append(
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": fname, "arguments": json.dumps(args)},
            }
        )
        followup_messages.append(
            {
                "role": "function",
                "name": fname,
                "content": json.dumps(function_result),
            }
        )

        try:
            second_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=followup_messages,
                temperature=0.2,
            )
            final_choice = second_response.choices[0].message
            final_content = (
                final_choice.get("content")
                if isinstance(final_choice, dict)
                else getattr(final_choice, "content", None)
            )
            return {
                "assistant_content": assistant_content,
                "function_call": {"name": fname, "arguments": args},
                "final_content": final_content,
                "raw": second_response,
                "function_result": function_result,  # CHANGED: include function_result for /chat inspection
            }
        except Exception as e:
            return {
                "assistant_content": assistant_content,
                "function_call": {"name": fname, "arguments": args},
                "final_content": json.dumps(function_result),
                "raw": {
                    "error_followup": str(e),
                    "function_result": function_result,
                },
                "function_result": function_result,  # CHANGED
            }

    # -------------------------------
    # Step 7: Return AI's plain content
    # -------------------------------
    if assistant_content is None:
        try:
            assistant_content = (
                raw_choice.get("content")
                if isinstance(raw_choice, dict)
                else getattr(raw_choice, "content", None)
            )
        except Exception:
            assistant_content = None

    return {
        "assistant_content": assistant_content,
        "function_call": None,
        "final_content": None,
        "raw": response,
        "function_result": None,  # CHANGED: always include for consistency
    }

# =========================================================
# STEP 8: POST /chat — Main Chat Logic (updated to call OpenAI first)
# - We call OpenAI for every message (requirement).
# - If the user is still in login steps (0-2) we preserve your original login logic exactly.
# - After login (login_step >= 3) we honor OpenAI's function_call or fallback to original parsing.
# =========================================================
@app.post("/chat")
async def chat(req: ChatRequest):
    msg = req.message.strip()
    sid = req.session_id

    # Ensure session exists (same initialization as your original code)
    if sid not in sessions:
        sessions[sid] = {
            "login_step": 0,
            "user_info": {},
            "cart": {},
            "items": [],
            "categories": get_categories(),
            "selected_cat": None,
            "payment_method": None,             # ADDED: store normalized payment method
            "awaiting_payment_method": False,   # ADDED: flag while waiting for payment reply
        }

    state = sessions[sid]
    reply = ""
    logout = False

    # -------------------------------
    # ADDED: If the bot is awaiting payment method, handle the next user message locally
    # -------------------------------
    if state.get("awaiting_payment_method"):
        # CHANGED: handle user's payment-method response before calling OpenAI
        normalized = parse_payment_method(msg)
        if not normalized:
            # couldn't understand user's choice — re-prompt
            reply = "❌ I didn't understand. Please reply with 'Cash on Delivery' or 'Online Transfer'."
            return {"reply": reply, "session": state}
        # save normalized method
        state["payment_method"] = normalized
        state["awaiting_payment_method"] = False
        # proceed to finalize checkout now that we have payment method
        function_result = fn_checkout(state, sid)
        if function_result.get("needs_payment_method"):
            # unexpected — re-prompt (defensive)
            state["awaiting_payment_method"] = True
            reply = function_result.get("message", "Do you want to pay via Cash on Delivery or Online Transfer?")
            return {"reply": reply, "session": state}
        if function_result.get("success"):
            # build friendly final bill reply (same style as other code paths)
            bill = function_result.get("bill", {})
            lines = []
            for l in bill.get("lines", []):
                lines.append(f"{l['name']}: {l['quantity']} x {l['price']} = {l['subtotal']} Rs")
            total = bill.get("total", 0)
            user = state.get("user_info", {})
            payment = function_result.get("paymentMethod", state.get("payment_method"))
            reply = (
                "🧾 Final Bill:\n" + "\n".join(lines) +
                f"\n\n💰 Total: {total} Rs\n👤 {user.get('name')}\n📞 {user.get('phone')}\n🏠 {user.get('address')}\n"
                f"💳 Payment method: {payment}"
            )
            logout = True
            return {"reply": reply, "session": state}
        else:
            reply = function_result.get("message", "Failed to complete checkout.")
            return {"reply": reply, "session": state}

    # -------------------------------
    # Step A: Send every message to OpenAI first (requirement)
    # -------------------------------
    openai_result = {}
    if state.get("login_step", 0) >= 3:
        openai_result = call_openai_and_maybe_execute(msg, state, sid)

    # Note: openai_result may contain "function_call" (intention to call backend),
    # "final_content" (if we executed a function and used OpenAI to craft final text),
    # or "assistant_content" (if OpenAI replied with plain text and did NOT call a function).
    #
    # We will now combine OpenAI's decision with your original flow:
    # - If user is in login flow (login_step < 3), keep original login processing (unchanged).
    # - Else (user logged-in), prefer OpenAI-driven function-calling if present; if not present,
    #   fall back to your original parsing for commands like "add X", "checkout", "bill".
    # -------------------------------

    # -------------------------------
    # Login Flow (unchanged behavior)
    # -------------------------------
    # -------------------------------
    # Login Flow
    # -------------------------------
    if state["login_step"] == 0:
        # Let AI handle asking + validating name
        system_context = (
            "You are a polite bilingual shopping assistant for Point of Sale.\n"
            "Currently you are in LOGIN MODE (step 0).\n"
            "Your job is to:\n"
            "1. Politely ask the user for their name if they greet (hi, hello, salam, etc.).\n"
            "2. Validate that the name contains only letters (English a-z or Urdu script).\n"
            "   - Reject numbers, symbols, random strings, or common greetings.\n"
            "3. Once a valid name is given, clearly confirm it with a ✅ message like:\n"
            "   '✅ Thanks [Name], saved your name.'\n"
            "   Do not proceed to phone step, backend will do that.\n"
            "4. IMPORTANT: Always reply in the language the user is using.\n"
            "   - If user speaks in English, reply in English.\n"
            "   - If user is speaking in Roman Urdu, reply in Roman Urdu.\n"
        )

        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": msg},
        ]

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
            )
            ai_reply = response.choices[0].message.content.strip()
        except Exception as e:
            ai_reply = f"⚠️ Error during login: {str(e)}"

        reply = ai_reply

        # Only move to phone step if AI has confirmed name with ✅
        if ai_reply.startswith("✅"):
            # Extract name after "✅ Thanks ..."
            extracted = ai_reply.replace("✅", "").replace("Thanks", "").replace("saved your name", "").strip()
            if extracted:
                state["user_info"]["name"] = extracted
                state["login_step"] = 1
                reply = "📞 Please enter your phone number (starting with 03):"

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
            reply = (
                "✅ You're logged in!\n\n"
                "Here’s how you can shop:\n"
                "1️⃣ Type a category name (e.g., Vegetables, Fruits).\n"
                "2️⃣ I will show you items with prices.\n"
                "3️⃣ You can add items like 'add 2 carrots' or 'please give me 1 apple'.\n"
                "4️⃣ Type 'bill' to see your cart.\n"
                "5️⃣ Type 'checkout' when you’re ready.\n\n"
                f"Available categories: {', '.join(cat_names)}"
            )

        else:
            reply = "❌ Address cannot be empty."


    # -------------------------------
    # After Login: Use OpenAI function-calling or original parsing as fallback
    # -------------------------------
    else:
        # First: if OpenAI produced a final_content because it executed a function,
        # prefer that final content as the reply (this is the function-calling flow).
        if openai_result.get("final_content") is not None:
            # OpenAI created a natural-language reply after executing a function
            reply = openai_result["final_content"]
            # CHANGED: If the function executed was 'checkout', check function_result for payment requirement
            fc = openai_result.get("function_call")
            if fc and fc.get("name") == "checkout":
                fr = openai_result.get("function_result", {})
                if fr.get("needs_payment_method"):
                    # set flag so next user message will be treated as payment method
                    state["awaiting_payment_method"] = True
                elif fr.get("success"):
                    logout = True

        # If OpenAI produced a function_call but we did NOT execute it because the user
        # was not logged in, or because the helper returned the function_call (login case),
        # or for some reason final_content is not present, we fall back to original parsing.
        elif openai_result.get("function_call") and openai_result.get("final_content") is None:
            # If a function_call was returned but not executed, attempt to execute now if login_step >=3.
            # (Normally execution already done in call_openai_and_maybe_execute for logged-in users.)
            fc = openai_result["function_call"]
            fname = fc.get("name")
            args = fc.get("arguments", {})
            # Only run if we are logged in (should be), else ignore
            if state.get("login_step", 0) >= 3 and fname in FUNCTION_IMPLEMENTATIONS:
                try:
                    function_result = FUNCTION_IMPLEMENTATIONS[fname](state, sid=sid, args=args)
                    # Provide a simple textual reply for function result (safe fallback)
                    if fname == "show_bill":
                        if function_result.get("empty"):
                            reply = "🛒 Your cart is empty."
                        else:
                            lines = []
                            total = function_result.get("total", 0)
                            for l in function_result.get("lines", []):
                                lines.append(f"- {l['name']}: {l['quantity']} x {l['price']} = {l['subtotal']} Rs")
                            reply = "🧾 Your Cart:\n" + "\n".join(lines) + f"\n\n💰 Total: {total} Rs"
                    elif fname == "checkout":
                        # CHANGED: If checkout indicates payment method required, set awaiting flag and prompt
                        if function_result.get("needs_payment_method"):
                            state["awaiting_payment_method"] = True
                            reply = function_result.get("message") or "Do you want to pay via Cash on Delivery or Online Transfer?"
                        elif function_result.get("success"):
                            # behave like original checkout reply
                            bill = function_result.get("bill", {})
                            lines = []
                            for l in bill.get("lines", []):
                                lines.append(f"{l['name']}: {l['quantity']} x {l['price']} = {l['subtotal']} Rs")
                            total = bill.get("total", 0)
                            user = state.get("user_info", {})
                            # CHANGED: include paymentMethod if present in function_result
                            payment = function_result.get("paymentMethod", state.get("payment_method"))
                            reply = f"🧾 Final Bill:\n" + "\n".join(lines) + f"\n\n💰 Total: {total} Rs\n👤 {user.get('name')}\n📞 {user.get('phone')}\n🏠 {user.get('address')}\n💳 Payment method: {payment}"
                            logout = True
                        else:
                            # For non-success, show returned message or raw structure
                            if isinstance(function_result, dict) and function_result.get("message"):
                                reply = function_result.get("message")
                            else:
                                reply = json.dumps(function_result)
                    else:
                        # For add_to_cart or get_items etc, try to create friendly messages
                        if isinstance(function_result, dict) and function_result.get("message"):
                            reply = function_result.get("message")
                        else:
                            reply = json.dumps(function_result)
                except Exception as e:
                    reply = "⚠️ Failed to execute requested action."
            else:
                # fallback to original parsing below
                openai_result = {"function_call": None}  # clear to go to fallback

        # If OpenAI did NOT call a function (most common), fall back to original parsing logic
        if not openai_result.get("final_content") and not openai_result.get("function_call"):
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

            # Add item (old style command 'add <qty> <item>' or 'add <item>')
            elif any(x in msg_lower for x in ["add", "give me", "i want", "please add"]):
                qty = 1
                item_name = ""

                # regex like "2 carrots" or "carrots 2"
                match = re.search(r'(\d+)\s+([a-zA-Z ]+)', msg_lower)
                if match:
                    qty = int(match.group(1))
                    item_name = match.group(2).strip()
                else:
                    # default: try last word(s)
                    parts = msg_lower.split()
                    for i, p in enumerate(parts):
                        if p.isdigit():
                            qty = int(p)
                            item_name = " ".join(parts[i + 1:])
                            break
                    if not item_name:
                        item_name = msg_lower.replace("add", "").replace("give me", "").replace("i want", "").strip()

                matched = _match_item_in_state(state, item_name)
                if matched:
                    reply = fn_add_to_cart(state, matched["itemName"], qty)["message"]
                else:
                    reply = f"❌ Item '{item_name}' not found."

            # Checkout (old-style)
            elif msg_lower == "checkout":
                cart = state["cart"]
                if not cart:
                    reply = "🛒 Your cart is empty."
                else:
                    # CHANGED: If no payment method saved, ask for it and set awaiting flag
                    if not state.get("payment_method"):
                        state["awaiting_payment_method"] = True
                        reply = "Do you want to pay via Cash on Delivery or Online Transfer?"
                    else:
                        # We have a payment method — finalize checkout as before
                        function_result = fn_checkout(state, sid)
                        if function_result.get("needs_payment_method"):
                            # Defensive: if local checkout still requests payment method
                            state["awaiting_payment_method"] = True
                            reply = function_result.get("message") or "Do you want to pay via Cash on Delivery or Online Transfer?"
                        elif function_result.get("success"):
                            bill = function_result.get("bill", {})
                            lines = []
                            for l in bill.get("lines", []):
                                lines.append(f"{l['name']}: {l['quantity']} x {l['price']} = {l['subtotal']} Rs")
                            total = bill.get("total", 0)
                            user = state.get("user_info", {})
                            payment = function_result.get("paymentMethod", state.get("payment_method"))
                            reply = f"🧾 Final Bill:\n" + "\n".join(lines) + f"\n\n💰 Total: {total} Rs\n👤 {user.get('name')}\n📞 {user.get('phone')}\n🏠 {user.get('address')}\n💳 Payment method: {payment}"
                            logout = True
                        else:
                            reply = function_result.get("message", "Failed to checkout.")

            # Category selection (old-style: user types the category name)
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
                    # As a final fallback: if OpenAI returned a plain assistant_content earlier, use it.
                    assistant_plain = openai_result.get("assistant_content")
                    if assistant_plain:
                        # Per requirement, if OpenAI considers it unrelated/general it returns exact phrase:
                        # "Sorry, I can only help you with shopping-related queries at Point of Sale."
                        reply = assistant_plain
                    else:
                        # last fallback: replicate previous behavior of handing to AI short response
                        # (but we already called OpenAI earlier). If nothing, give a safe fallback.
                        reply = "⚠️ I didn't understand that. You can type a category name or 'bill' or 'checkout'."

    # -------------------------------
    # Auto logout / cleanup when logout was triggered (same as original)
    # -------------------------------
    if logout:
        try:
            # CHANGED: include paymentMethod in logout POST
            requests.post(BILL_API_URL, json={
                "user": state["user_info"],
                "cart": state["cart"],
                "paymentMethod": state.get("payment_method")
            })
        except:
            pass
        threading.Thread(target=reset_session, args=(sid,), daemon=True).start()

    return {"reply": reply, "session": state}

# =========================================================
# STEP 9: Serve index.html (unchanged)
# =========================================================
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")
