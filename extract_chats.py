import json
import os
from datetime import datetime

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_text(parts):
    extracted = []
    for part in parts:
        if isinstance(part, str):
            extracted.append(part)
        elif isinstance(part, dict):
            # Handle different content types gracefully
            if "text" in part:
                extracted.append(part["text"])
            elif "message_type" in part:
                extracted.append(f"[{part['message_type']}]")
            else:
                extracted.append(str(part))
        else:
            extracted.append(str(part))
    return "\n".join(extracted).strip()

def get_ordered_messages(mapping, root_id):
    messages = []
    node_id = root_id
    while node_id:
        node = mapping.get(node_id)
        if not node:
            break
        msg = node.get("message")
        if msg:
            role = msg.get("author", {}).get("role", "system")
            parts = msg.get("content", {}).get("parts", [])
            text = extract_text(parts)
            timestamp = msg.get("create_time", None)
            messages.append((role, text, timestamp))
        # Move to next child (assuming linear convo, so one child)
        children = node.get("children", [])
        node_id = children[0] if children else None
    return messages

def write_convo_md(convo, output_dir):
    title = convo.get("title", "untitled")
    mapping = convo.get("mapping", {})
    root_id = [k for k,v in mapping.items() if v.get("parent") is None][0]
    messages = get_ordered_messages(mapping, root_id)
    
    # Get date from conversation create_time
    dt = datetime.fromtimestamp(convo.get("create_time", datetime.utcnow().timestamp()))
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    folder = os.path.join(output_dir, year, month)
    ensure_dir(folder)
    
    filename = f"{dt.strftime('%Y%m%d_%H%M%S')}_{title.replace(' ', '_')[:30]}.md"
    path = os.path.join(folder, filename)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n")
        f.write(f"*Created: {dt.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        for role, text, ts in messages:
            if not text.strip():
                continue
            ts_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else "Unknown"
            f.write(f"**{role.capitalize()} ({ts_str})**\n\n{text.strip()}\n\n---\n")
    print(f"✅ Saved: {path}")

def convert_conversations(convos_path, output_dir):
    with open(convos_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for convo in data:
        write_convo_md(convo, output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert ChatGPT export conversations.json to Markdown.")
    parser.add_argument("json_path", help="Path to conversations.json")
    parser.add_argument("output_dir", help="Where to save the Markdown files")
    args = parser.parse_args()
    
    convert_conversations(args.json_path, args.output_dir)