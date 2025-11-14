"""
프로젝트 구조 확인 스크립트
"""

import os


def print_tree(directory, prefix="", max_depth=3, current_depth=0):
    """디렉토리 구조를 트리 형태로 출력"""
    if current_depth >= max_depth:
        return
    
    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        return
    
    # __pycache__ 제외
    entries = [e for e in entries if e != '__pycache__']
    
    for i, entry in enumerate(entries):
        path = os.path.join(directory, entry)
        is_last = i == len(entries) - 1
        
        # 트리 구조 문자
        connector = "└── " if is_last else "├── "
        
        # 디렉토리인지 확인
        if os.path.isdir(path):
            print(f"{prefix}{connector}{entry}/")
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension, max_depth, current_depth + 1)
        else:
            # 파일 크기 표시
            size = os.path.getsize(path)
            size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
            print(f"{prefix}{connector}{entry}  ({size_str})")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("프로젝트 구조")
    print("="*70 + "\n")
    
    current_dir = os.getcwd()
    print(f"📁 {os.path.basename(current_dir)}/\n")
    
    print_tree(current_dir, max_depth=3)
    
    print("\n" + "="*70)
    print("파일 개수 통계")
    print("="*70)
    
    # 확장자별 통계
    extensions = {}
    total_files = 0
    
    for root, dirs, files in os.walk(current_dir):
        # __pycache__ 제외
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            total_files += 1
            ext = os.path.splitext(file)[1] or 'no extension'
            extensions[ext] = extensions.get(ext, 0) + 1
    
    print(f"\n총 파일 수: {total_files}")
    print("\n확장자별 분류:")
    for ext, count in sorted(extensions.items(), key=lambda x: -x[1]):
        print(f"  {ext:15s}: {count:3d} 개")
    
    print("\n" + "="*70 + "\n")
