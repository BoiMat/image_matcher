import os
import requests
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import cv2
import numpy as np
from PIL import Image
import io
from matching import get_matcher, available_models
from matching.viz import *
import torch

# IIIF image endpoint
IIIF_BASE = "https://www.e-rara.ch/i3f/v21/{record_id}/full/{size}/0/default.jpg"
# Manifest endpoint to get all pages for a document
MANIFEST_URL = "https://www.e-rara.ch/i3f/v21/{record_id}/manifest"

def get_image_url(record_id, size="full"):
    """
    Generate the IIIF URL for an image
    
    Parameters:
    -----------
    record_id : str
        The e-rara record ID
    size : str, optional
        Size parameter for IIIF (default: "full")
        Can be: "full", "max", ",150" (for thumbnails), etc.
    
    Returns:
    --------
    str
        Complete IIIF URL for the image
    """
    return IIIF_BASE.format(record_id=record_id, size=size)


def get_manifest_url(record_id):
    """
    Generate the IIIF manifest URL for a record
    """
    return MANIFEST_URL.format(record_id=record_id)

def extract_page_ids_from_manifest(manifest):
    """
    Extract all page IDs from a IIIF manifest
    
    Parameters:
    -----------
    manifest : dict
        The IIIF manifest JSON
    
    Returns:
    --------
    list
        List of page IDs
    """
    page_ids = []
    
    sequences = manifest.get('sequences', [])
    for sequence in sequences:
        canvases = sequence.get('canvases', [])
        for canvas in canvases:
            images = canvas.get('images', [])
            for image in images:
                resource = image.get('resource', {})
                service = resource.get('service', {})
                if '@id' in service:
                    service_id = service['@id']
                    # Extract the ID from the format: https://www.e-rara.ch/i3f/v21/13465786
                    if '/i3f/v21/' in service_id:
                        page_id = service_id.split('/i3f/v21/')[1]
                        page_ids.append(page_id)
    
    return page_ids

def get_all_page_ids(record_id, timeout=30, retries=3):
    """
    Get all page IDs for a given record by fetching its IIIF manifest
    
    Parameters:
    -----------
    record_id : str
        The e-rara record ID
    timeout : int, optional
        Request timeout in seconds
    retries : int, optional
        Number of retry attempts
    
    Returns:
    --------
    dict
        Dictionary with manifest data and page IDs
    """
    manifest_url = get_manifest_url(record_id)
    
    for attempt in range(retries):
        try:
            response = requests.get(manifest_url, timeout=timeout)
            response.raise_for_status()
            
            manifest = response.json()
            
            page_ids = extract_page_ids_from_manifest(manifest)
            
            metadata = {}
            if 'label' in manifest:
                metadata['title'] = manifest['label']
                
            if 'metadata' in manifest:
                for item in manifest['metadata']:
                    if 'label' in item and 'value' in item:
                        metadata[item['label']] = item['value']
            
            return {
                'record_id': record_id,
                'manifest': manifest_url,
                'page_ids': page_ids,
                'page_count': len(page_ids),
                'metadata': metadata
            }
            
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            else:
                return {
                    'record_id': record_id,
                    'error': str(e),
                    'page_ids': []
                }
    
    return {
        'record_id': record_id,
        'error': f"Failed to fetch manifest after {retries} attempts",
        'page_ids': []
    }

def get_all_page_ids_from_records(record_ids, max_workers=5, timeout=30, retries=3):
    """
    Get all page IDs from multiple records
    
    Parameters:
    -----------
    record_ids : list
        List of e-rara record IDs
    max_workers : int, optional
        Maximum number of parallel requests
    timeout : int, optional
        Request timeout in seconds
    retries : int, optional
        Number of retry attempts
        
    Returns:
    --------
    list
        List of all page IDs from all records
    """
    all_page_ids = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_all_page_ids, rec_id, timeout, retries) for rec_id in record_ids]
        
        for future in tqdm(futures, desc="Fetching manifests", unit="record"):
            try:
                result = future.result()
                page_ids = result.get('page_ids', [])
                all_page_ids.extend(page_ids)
            except Exception as e:
                print(f"Error processing record: {e}")
    
    return all_page_ids

def bytes2tensor(image_bytes):
    """
    Convert image bytes to a PyTorch tensor
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Change to CxHxW format
    return torch.tensor(image)

def bytes2cv2(image_bytes):
    """
    Convert image bytes to a cv2 image
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = np.array(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def aggressive_border_removal(img: np.ndarray,
                              border_threshold: int = 50,
                              crop_percentage: float = 0.05) -> np.ndarray:
    """
    More aggressive border removal that crops a percentage from each side,
    then further trims any mostly‐black borders via thresholding.
    Works on NumPy arrays (grayscale H×W or color H×W×C), returning the same format.
    
    Parameters:
        img               Input image as a NumPy array (uint8). If color, assumed RGB order.
        border_threshold  Intensity threshold for binary‐masking (0–255) to detect 'content'.
        crop_percentage   Fractional amount to initially crop on each side (e.g. 0.05 for 5%).
    
    Returns:
        Cropped NumPy array (same dtype and number of channels as input).
    """
    if img.ndim == 3 and img.shape[2] == 3:
        is_color = True
        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2:
        is_color = False
        h, w = img.shape
        gray = img
    else:
        raise ValueError("Expected input to be either H×W (grayscale) or H×W×3 (RGB).")
    
    crop_h = int(h * crop_percentage)
    crop_w = int(w * crop_percentage)

    if is_color:
        cropped = img[crop_h : h - crop_h, crop_w : w - crop_w, :]
        cropped_gray = gray[crop_h : h - crop_h, crop_w : w - crop_w]
    else:
        cropped = img[crop_h : h - crop_h, crop_w : w - crop_w]
        cropped_gray = cropped

    _, binary = cv2.threshold(cropped_gray,
                              border_threshold,
                              255,
                              cv2.THRESH_BINARY)

    row_sums = np.sum(binary, axis=1)
    col_sums = np.sum(binary, axis=0)

    cropped_h, cropped_w = cropped_gray.shape[:2]

    row_threshold = cropped_w * 20
    col_threshold = cropped_h * 20

    valid_rows = np.where(row_sums > row_threshold)[0]
    valid_cols = np.where(col_sums > col_threshold)[0]

    if valid_rows.size == 0 or valid_cols.size == 0:
        return cropped

    y1, y2 = valid_rows[0], valid_rows[-1]
    x1, x2 = valid_cols[0], valid_cols[-1]

    buffer = 5
    y1 = max(0, y1 - buffer)
    x1 = max(0, x1 - buffer)
    y2 = min(cropped_h, y2 + buffer)
    x2 = min(cropped_w, x2 + buffer)

    if is_color:
        return cropped[y1:y2, x1:x2, :]
    else:
        return cropped[y1:y2, x1:x2]


def load_target_image(target_image_path, size=",300", preprocessing=False):
    """
    Load the target image, resize it based on height, and convert to tensor
    
    Parameters:
    -----------
    target_image_path : str
        Path to the target image file
    size : str, optional
        Size parameter for height (default: ",300" for 300px height)
        Can be: ",300", "400", "full" (no resize)
    preprocessing : bool, optional
        Whether to apply aggressive border removal (default: False)
        
    Returns:
    --------
    torch.Tensor
        Preprocessed image tensor in CxHxW format
    """
    if not os.path.exists(target_image_path):
        raise FileNotFoundError(f"Target image file not found: {target_image_path}")

    img = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image file: {target_image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if size.startswith(","):
        target_height = int(size[1:])
        current_height, current_width = img.shape[:2]
        
        if current_height != target_height:
            aspect_ratio = current_width / current_height
            new_width = int(target_height * aspect_ratio)
            img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
            
    elif size == "full":
        pass
    else:
        target_height = int(size)
        current_height, current_width = img.shape[:2]
        
        if current_height != target_height:
            aspect_ratio = current_width / current_height
            new_width = int(target_height * aspect_ratio)
            img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
    
    if preprocessing:
        img = aggressive_border_removal(img, crop_percentage=0.05)

    # Convert to tensor: HxWxC -> CxHxW, normalize to [0,1]
    img = img.astype(np.float32) / 255.0
    torch_img = torch.tensor(img).permute(2, 0, 1)
    
    return torch_img

def download_tensor_img(record_id, size=",300", timeout=30, retries=3, preprocessing=True):
    """
    Download a thumbnail image and extract SIFT features WITHOUT saving to disk
    
    Parameters:
    -----------
    record_id : str
        The e-rara record ID
    size : str, optional
        Size parameter for IIIF (default: "300," for better SIFT extraction)
    timeout : int, optional
        Request timeout in seconds
    retries : int, optional
        Number of retry attempts
    max_features : int, optional
        Maximum number of SIFT features to extract
        
    Returns:
    --------
    dict
        Dictionary containing success status, features, and metadata
    """
    url = get_image_url(record_id, size)
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            numpy_img = bytes2cv2(response.content)
            
            if numpy_img is None:
                return False, f"Failed to decode image for {record_id}"

            numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)

            if preprocessing:
                numpy_img = aggressive_border_removal(numpy_img, crop_percentage=0.05)
            
            if numpy_img is None or numpy_img.size == 0:
                return False, f"Failed to process image for {record_id}: empty or invalid image data"
            
            numpy_img = numpy_img.astype(np.float32) / 255.0
            torch_img = torch.tensor(numpy_img).permute(2, 0, 1)
            return True, torch_img
            
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            else:
                return False, f"Error downloading {record_id}: {str(e)}"
        except Exception as e:
            return False, f"Error processing image for {record_id}: {str(e)}"
    
    return False, f"Failed to download {record_id} after {retries} attempts"
    
def download_full_image(record_id, output_dir, size="full", timeout=30, retries=3):
    """
    Download a full-resolution image and save to disk
    
    Parameters:
    -----------
    record_id : str
        The e-rara record ID
    output_dir : str
        Directory to save the image
    size : str, optional
        Size parameter for IIIF (default: "full")
    timeout : int, optional
        Request timeout in seconds
    retries : int, optional
        Number of retry attempts
        
    Returns:
    --------
    bool
        True if download was successful, False otherwise
    str
        Path to the downloaded image or error message
    """
    url = get_image_url(record_id, size)
    filename = f"{record_id}.jpg"
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        return True, output_path
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)
                
            return True, output_path
            
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            else:
                return False, f"Error downloading {record_id}: {str(e)}"
    
    return False, f"Failed to download {record_id} after {retries} attempts"

def filter_by_target_similarity(record_ids, target_image, matcher='superpoint-lg', device='cpu', max_workers=5, 
                                size=",300", timeout=30, retries=3, preprocessing=True, inliers_threshold=100,
                                progress_callback=None):
    """
    Filter images based on Matcher inliers with parallel processing and better error handling
    """
    matcher_instance = get_matcher(matcher, device=device)

    if matcher_instance is None:
        return {
            'error': f"Matcher '{matcher}' is not available. Available models: {available_models}"
        }

    def _process_single_record(record_id):
        """Process a single record - download, preprocess, and match"""
        try:
            success, tensor_img = download_tensor_img(record_id, size, timeout, retries, preprocessing)
            
            if not success:
                return {
                    'record_id': record_id,
                    'success': False,
                    'error': tensor_img,  # tensor_img contains the error message
                    'good_matches': 0,
                    'url': get_image_url(record_id, size),
                    'thumbnail_size': 0
                }
            
            if target_image.device != tensor_img.device:
                tensor_img = tensor_img.to(target_image.device)
            
            try:
                result = matcher_instance(target_image, tensor_img)
                num_inliers = result.get('num_inliers', 0)
                
                return {
                    'record_id': record_id,
                    'success': True,
                    'good_matches': num_inliers,
                    'url': get_image_url(record_id, size),
                    'thumbnail_size': tensor_img.element_size() * tensor_img.nelement(),
                    'meets_threshold': num_inliers >= inliers_threshold
                }
                    
            except Exception as matcher_error:
                return {
                    'record_id': record_id,
                    'success': False,
                    'error': f"Matcher error: {str(matcher_error)}",
                    'good_matches': 0,
                    'url': get_image_url(record_id, size),
                    'thumbnail_size': 0
                }
                
        except Exception as e:
            return {
                'record_id': record_id,
                'success': False,
                'error': f"Unexpected error: {str(e)}",
                'good_matches': 0,
                'url': get_image_url(record_id, size),
                'thumbnail_size': 0
            }

    print(f"Processing {len(record_ids)} images in parallel with {max_workers} workers...")
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_single_record, record_id) for record_id in record_ids]
        
        # Replace the tqdm loop with manual progress tracking
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(i + 1, len(futures), result.get('record_id', 'unknown'))
                
            except Exception as e:
                results.append({
                    'record_id': 'unknown',
                    'success': False,
                    'error': f"Future execution error: {str(e)}",
                    'good_matches': 0,
                })
                
                # Call progress callback for errors too
                if progress_callback:
                    progress_callback(i + 1, len(futures), 'error')

    return results

def download_selected_candidates(candidates, output_dir, size="full", max_workers=5, timeout=30, retries=3):
    """
    Download ONLY the selected candidate images at full resolution
    
    Parameters:
    -----------
    candidates : list
        List of candidate dictionaries from filter_by_target_similarity
    output_dir : str
        Directory to save the images
    size : str, optional
        Size parameter for IIIF (default: "full")
    max_workers : int, optional
        Maximum number of parallel downloads
    timeout : int, optional
        Request timeout in seconds
    retries : int, optional
        Number of retry attempts
        
    Returns:
    --------
    dict
        Dictionary with download results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    download_results = {
        "successful": 0,
        "failed": 0,
        "errors": [],
        "downloaded_files": []
    }
    
    def _download_candidate(candidate):
        record_id = candidate['record_id']
        success, result = download_full_image(record_id, output_dir, size, timeout, retries)
        
        if success:
            return {
                'success': True,
                'record_id': record_id,
                'file_path': result,
                'good_matches': candidate['good_matches']
            }
        else:
            return {
                'success': False,
                'record_id': record_id,
                'error': result,
                'good_matches': candidate['good_matches']
            }
    
    print(f"Downloading {len(candidates)} selected candidates at full resolution...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_candidate, candidate) for candidate in candidates]
        
        for future in tqdm(futures, desc="Downloading full images", unit="image"):
            try:
                result = future.result()
                if result['success']:
                    download_results["successful"] += 1
                    download_results["downloaded_files"].append(result)
                else:
                    download_results["failed"] += 1
                    download_results["errors"].append(result)
            except Exception as e:
                download_results["failed"] += 1
                download_results["errors"].append(f"Unexpected error: {str(e)}")
    
    return download_results

def batch_download_with_target_filtering(record_ids, target_image_path, output_dir, matcher='superpoint-lg',
                                         expand_to_pages=True, max_workers=5, device='cpu', size=",300",
                                         timeout=30, retries=3, preprocessing=False, min_matches=100,
                                         progress_callback=None):
    """
    Complete pipeline: filter images based on matcher inliers and download ONLY candidates
    
    Parameters:
    -----------
    record_ids : list
        List of e-rara record IDs
    target_image_path : str
        Path to the target image file
    output_dir : str
        Directory to save the images
    min_matches : int, optional
        Minimum number of good SIFT matches required
    min_similarity_score : float, optional
        Minimum similarity score required
    max_candidates : int, optional
        Maximum number of candidates to download
    expand_to_pages : bool, optional
        Whether to expand record IDs to individual page IDs
    size : str, optional
        Size parameter for IIIF (default: "full")
    max_workers : int, optional
        Maximum number of parallel downloads
    timeout : int, optional
        Request timeout in seconds
    retries : int, optional
        Number of retry attempts
    max_features : int, optional
        Maximum number of SIFT features to extract per image
        
    Returns:
    --------
    dict
        Dictionary with complete results
    """
    try:
        target_image = load_target_image(target_image_path, size, preprocessing=False)
    except Exception as e:
        return {'error': f"Failed to load target image: {str(e)}"}

    if expand_to_pages:
        print("Expanding record IDs to individual page IDs...")
        all_page_ids = get_all_page_ids_from_records(record_ids, max_workers, timeout, retries)
        ids_to_process = all_page_ids
        print(f"Expanded {len(record_ids)} records to {len(all_page_ids)} pages")
    else:
        ids_to_process = record_ids
        print(f"Processing {len(ids_to_process)} record IDs directly")

    filtering_results = filter_by_target_similarity(
        ids_to_process, target_image, matcher=matcher, device=device, 
        size=size, timeout=timeout, retries=retries, max_workers=max_workers,
        preprocessing=preprocessing, inliers_threshold=min_matches,
        progress_callback=progress_callback  # Pass the callback through
    )

    if 'error' in filtering_results:
        return filtering_results
    
    successful_candidates = [r for r in filtering_results if r['success'] and r.get('meets_threshold', False)]
    
    print(f"\nProcessing complete:")
    print(f"  Total processed: {len(filtering_results)}")
    print(f"  Successful matches: {len([r for r in filtering_results if r['success']])}")
    print(f"  Candidates above threshold: {len(successful_candidates)}")
    
    if not successful_candidates:
        print("No similar candidates found.")
        return {
            'filtering': filtering_results,
            'download': {'successful': 0, 'failed': 0, 'downloaded_files': []},
            'summary': {
                'input_images': len(ids_to_process),
                'candidates_found': 0,
                'successfully_downloaded': 0,
                'target_image': target_image_path,
                'data_saved': 'Significant - only thumbnails processed, no full images downloaded'
            }
        }

    successful_candidates.sort(key=lambda x: x['good_matches'], reverse=True)
    print(f"  Best match: {successful_candidates[0]['good_matches']} inliers")
    print(f"  Worst match: {successful_candidates[-1]['good_matches']} inliers")

    download_results = download_selected_candidates(successful_candidates, output_dir, size, max_workers, timeout, retries)
    
    total_thumbnail_size = sum(r.get('thumbnail_size', 0) for r in filtering_results if r['success'] and r.get('meets_threshold', False))
    
    final_results = {
        'summary': {
            'input_images': len(ids_to_process),
            'candidates_found': len(successful_candidates),
            'successfully_downloaded': download_results['successful'],
            'target_image': target_image_path,
            'min_matches': min_matches,
            'data_efficiency': {
                'thumbnail_data_downloaded_mb': total_thumbnail_size / (1024 * 1024),
                'reduction_ratio': 1 - (len(successful_candidates) / len(ids_to_process)) if ids_to_process else 0
            }
        },
        'filtering': filtering_results,
        'download': download_results,
    }
    
    results_path = os.path.join(output_dir, "filtering_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description='Download images from e-rara filtered by SIFT similarity to target image')
    parser.add_argument('ids_file', help='File containing record IDs (one per line) or JSON file with "ids" key')
    parser.add_argument('target_image', help='Path to target image file for similarity comparison')
    parser.add_argument('--output-dir', default='e_rara_candidates', help='Output directory for downloaded images')
    parser.add_argument('--matcher', default='superpoint-lg', help='Matcher model to use (default: superpoint-lg)')
    parser.add_argument('--device', type=str, default=None, help='Device to run the matcher on (default: cpu)')
    parser.add_argument('--preprocessing', type=bool, default=False, help='Apply aggressive border removal to target image (default: False)')
    parser.add_argument('--size', default=',300', help='Size parameter for IIIF (default: full)')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of parallel downloads')
    parser.add_argument('--min-matches', type=int, default=100, 
                        help='Minimum number of matches required (superpoint-lg: 100, sift-nn: 20)')
    parser.add_argument('--no-expand', action='store_true',
                        help='Do not expand record IDs to individual pages (use record IDs directly)')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds')
    parser.add_argument('--retries', type=int, default=3, help='Number of retry attempts')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.target_image):
        print(f"Error: Target image file not found: {args.target_image}")
        return
    
    if args.device is None:
        args.device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        print(f"No device specified, using default: {args.device}")

    if args.device not in ['cpu', 'cuda']:
        print(f"Error: Invalid device '{args.device}'. Must be 'cpu' or 'cuda'.")
        return
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA device requested but no GPU available. Falling back to CPU.")
        args.device = 'cpu'

    if args.device == 'cpu' and args.matcher == 'superpoint-lg':
        print("Warning: Using CPU with superpoint-lg matcher may be slow. Consider using a GPU if available or switch to 'sift-nn' matcher.")
    
    if 'sift' in args.matcher and args.min_matches == 100:
        print("Warning: Using SIFT matcher but default min_matches is set to 100. Automatically setting to 20 for 'sift-nn' matcher.")
        args.min_matches = 20
    
    record_ids = []
    try:
        if args.ids_file.endswith('.json'):
            with open(args.ids_file, 'r') as f:
                data = json.load(f)
                record_ids = data.get('ids', [])
        else:
            with open(args.ids_file, 'r') as f:
                record_ids = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading record IDs: {str(e)}")
        return
    
    print(f"Loaded {len(record_ids)} record IDs")
    print(f"Target image: {args.target_image}")
    print("Strategy: Download thumbnails → Compare with target → Download candidates only")
    
    results = batch_download_with_target_filtering(
        record_ids=record_ids,
        target_image_path=args.target_image,
        output_dir=args.output_dir,
        matcher=args.matcher,
        device=args.device,
        min_matches=args.min_matches,
        expand_to_pages=not args.no_expand,  # expand_to_pages is opposite of no_expand
        size=args.size,
        max_workers=args.max_workers,
        timeout=args.timeout,
        retries=args.retries,
        preprocessing=args.preprocessing
    )

    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"\nDownload and matching complete!")
    print(f"Input images: {results['summary']['input_images']}")
    print(f"Candidates found: {results['summary']['candidates_found']}")
    print(f"Successfully downloaded: {results['summary']['successfully_downloaded']}")
    print(f"Min matches threshold: {results['summary']['min_matches']}")
    
    if 'data_efficiency' in results['summary']:
        efficiency = results['summary']['data_efficiency']
        print(f"\nData Efficiency:")
        print(f"Thumbnail data processed: {efficiency['thumbnail_data_downloaded_mb']:.1f} MB")
        print(f"Reduction ratio: {efficiency['reduction_ratio']:.1%}")
    
    if results['download']['failed'] > 0:
        print(f"\nFailed downloads: {results['download']['failed']}")
        for error in results['download']['errors'][:5]:
            print(f"- {error}")
    
    if results['summary']['successfully_downloaded'] > 0:
        print(f"\nBest matches (by number of inliers):")
        for i, file_info in enumerate(results['download']['downloaded_files'][:5]):
            print(f"{i+1}. {file_info['record_id']} ({file_info['good_matches']} matches)")
    
    print(f"\nDetailed results saved to: {os.path.join(args.output_dir, 'filtering_results.json')}")

if __name__ == "__main__":
    main()