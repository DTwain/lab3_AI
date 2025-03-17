import os
import requests
import json
import time
import argparse

def get_repo_tree(owner, repo, branch='main'):
    """
    Retrieves the full repository tree (recursive) for the specified branch using GitHub's API.
    """
    # Get branch info to extract the commit's tree SHA.
    branch_url = f"https://api.github.com/repos/{owner}/{repo}/branches/{branch}"
    response = requests.get(branch_url)
    response.raise_for_status()
    branch_data = response.json()
    tree_sha = branch_data["commit"]["commit"]["tree"]["sha"]
    
    # Get the recursive tree listing.
    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{tree_sha}?recursive=1"
    response = requests.get(tree_url)
    response.raise_for_status()
    return response.json()

def is_image_file(file_path):
    """
    Checks if the file path ends with a common image extension.
    """
    return file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', 'webp'))

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve all image URLs from a GitHub repository, process them with OCR, and output a JSON file."
    )
    parser.add_argument("owner", help="GitHub repository owner (username or organization)")
    parser.add_argument("repo", help="GitHub repository name")
    parser.add_argument("--branch", default="main", help="Repository branch to use (default: main)")
    parser.add_argument("--output", default="data.json", help="Output JSON file (default: data.json)")
    args = parser.parse_args()

    # Retrieve the repository tree.
    print(f"Fetching repository tree for {args.owner}/{args.repo} on branch '{args.branch}'...")
    tree_data = get_repo_tree(args.owner, args.repo, args.branch)
    
    # Filter for image files.
    image_files = [
        item['path'] for item in tree_data.get('tree', [])
        if item['type'] == 'blob' and is_image_file(item['path'])
    ]
    print(f"Found {len(image_files)} image file(s).")
    
    results = []
    for file_path in image_files:
        # Construct the raw URL for the image.
        image_url = f"https://raw.githubusercontent.com/{args.owner}/{args.repo}/{args.branch}/{file_path}"
        print(f"Processing {image_url} ...")
        results.append({
            "image_url": image_url,
            "ground_truth": [],
            "ground_truth_boxes": []
        })
    
    # Write the results to the specified JSON file.
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results written to {args.output}")

if __name__ == '__main__':
    main()
