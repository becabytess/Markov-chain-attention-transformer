import os
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

raw_dir = r"c:\Users\beca\Desktop\gravimem-revived\data\raw\Takeout\YouTube and YouTube Music\history"
search_file = os.path.join(raw_dir, "search-history.html")
watch_file = os.path.join(raw_dir, "watch-history.html")

def parse_search_history(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    cells = re.findall(r'<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">(.*?)</div>', content, re.DOTALL)
    
    parsed = []
    for cell in cells:
        q_match = re.search(r'Searched for\s+<a[^>]*>(.*?)</a>', cell, re.DOTALL)
        if not q_match:
            q_match = re.search(r'Searched for\s+(.*?)(?:<br>|\n|$)', cell, re.DOTALL)
        
        text_lines = [re.sub(r'<[^>]+>', '', line).strip() for line in cell.split('<br>') if line.strip()]
        
        query = None
        date_str = None
        if q_match:
            query = re.sub(r'<[^>]+>', '', q_match.group(1)).strip()
        elif text_lines:
            query = text_lines[0]
            
        if len(text_lines) > 1:
            date_str = text_lines[-1]
            
        if query:
            query = re.sub(r'^Searched for\s+', '', query).strip()
            parsed.append({
                "type": "search",
                "query": query,
                "date_str": date_str
            })
    return parsed

def parse_watch_history(filepath, limit=1000):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        
    cells = re.findall(r'<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">(.*?)</div>', content, re.DOTALL)
    parsed = []
    for cell in cells:
        w_match = re.search(r'Watched\s+<a[^>]*>(.*?)</a>', cell, re.DOTALL)
        text_lines = [re.sub(r'<[^>]+>', '', line).strip() for line in cell.split('<br>') if line.strip()]
        
        title = None
        channel = None
        date_str = None
        
        if w_match:
            title = re.sub(r'<[^>]+>', '', w_match.group(1)).strip()
        elif text_lines:
            title = text_lines[0]
            
        if len(text_lines) >= 3:
            channel = text_lines[1]
            date_str = text_lines[2]
        elif len(text_lines) == 2:
            date_str = text_lines[1]
            
        if title:
            title = re.sub(r'^Watched\s+', '', title).strip()
            parsed.append({
                "type": "watch",
                "title": title,
                "channel": channel,
                "date_str": date_str
            })
    return parsed

print("--- PARSING SEARCH HISTORY ---")
searches = parse_search_history(search_file)
print(f"Total Search queries found: {len(searches)}")

print("\n--- SAMPLE SEARCH QUERIES (Most Recent) ---")
for i, s in enumerate(searches[:10]):
    print(f"[{i+1}] Query: \"{s['query']}\" | Date: {s['date_str']}")

print("\n--- SAMPLE SEARCH QUERIES (Oldest) ---")
for i, s in enumerate(searches[-5:]):
    print(f"[{len(searches)-5+i+1}] Query: \"{s['query']}\" | Date: {s['date_str']}")

print("\n--- PARSING WATCH HISTORY ---")
watches = parse_watch_history(watch_file)
print(f"Total Videos watched found: {len(watches)}")

print("\n--- SAMPLE WATCH HISTORY (Most Recent) ---")
for i, w in enumerate(watches[:10]):
    print(f"[{i+1}] Video: \"{w['title']}\" | Channel: {w['channel']} | Date: {w['date_str']}")
