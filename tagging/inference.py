import torch

def extract_triplets_from_tags(tag_table, id_to_sentiment, version='3D', 
                                confidence_scores=None, 
                                min_aspect_confidence=0.7, 
                                min_opinion_confidence=0.7,
                                min_sentiment_confidence=0.6):
    """
    Decodes a tag table from the model into a list of aspect-sentiment-opinion triplets.
    
    This function implements a confidence-based inference algorithm to find the most likely
    triplets based on the model's output grid and confidence scores.

    Args:
        tag_table (list of lists): The 2D grid of predicted tag IDs from the model.
        id_to_sentiment (dict): A mapping from sentiment ID to sentiment string (e.g., {1: 'POS'}).
        version (str, optional): The encoding version of the tags ('1D' or '3D'). Defaults to '3D'.
        confidence_scores (dict, optional): Dictionary containing confidence scores:
            - 'aspect': 2D tensor of aspect prediction confidences
            - 'opinion': 2D tensor of opinion prediction confidences
            - 'sentiment': 2D tensor of sentiment prediction confidences
        min_aspect_confidence (float): Minimum confidence threshold for aspects (default: 0.7)
        min_opinion_confidence (float): Minimum confidence threshold for opinions (default: 0.7)
        min_sentiment_confidence (float): Minimum confidence threshold for sentiments (default: 0.6)

    Returns:
        dict: A dictionary containing lists of found 'aspects', 'opinions', and 'triplets'.
    """
    tag_table_tensor = torch.tensor(tag_table)

    # Step 1: Decode the bitmask tags into separate boolean masks
    if version == '1D':  # {N, NEG, NEU, POS, O, A}
        is_aspect_span = (tag_table_tensor == 5) > 0
        is_opinion_span = (tag_table_tensor == 4) > 0
        sentiment_grid = tag_table_tensor * ((tag_table_tensor > 0) & (tag_table_tensor < 4))
    else:  # 3D: {N,A} - {N,O} - {N, NEG, NEU, POS}
        is_aspect_span = (tag_table_tensor & 8) > 0
        is_opinion_span = (tag_table_tensor & 4) > 0
        sentiment_grid = (tag_table_tensor & 3)

    # Step 2: Apply confidence filtering if confidence scores are provided
    if confidence_scores is not None:
        aspect_conf = confidence_scores.get('aspect', None)
        opinion_conf = confidence_scores.get('opinion', None)
        sentiment_conf = confidence_scores.get('sentiment', None)
        
        # Filter aspects by confidence
        if aspect_conf is not None:
            aspect_conf_tensor = torch.tensor(aspect_conf) if not isinstance(aspect_conf, torch.Tensor) else aspect_conf
            is_aspect_span = is_aspect_span & (aspect_conf_tensor >= min_aspect_confidence)
        
        # Filter opinions by confidence
        if opinion_conf is not None:
            opinion_conf_tensor = torch.tensor(opinion_conf) if not isinstance(opinion_conf, torch.Tensor) else opinion_conf
            is_opinion_span = is_opinion_span & (opinion_conf_tensor >= min_opinion_confidence)
        
        # Filter sentiments by confidence
        if sentiment_conf is not None:
            sentiment_conf_tensor = torch.tensor(sentiment_conf) if not isinstance(sentiment_conf, torch.Tensor) else sentiment_conf
            sentiment_grid = sentiment_grid * (sentiment_conf_tensor >= min_sentiment_confidence)

    # Step 3: Identify all spans that have a sentiment label
    sentiment_span_indices = sentiment_grid.nonzero()
    sentiment_values = sentiment_grid[sentiment_span_indices[:, 0], sentiment_span_indices[:, 1]].unsqueeze(dim=-1)
    
    # Create a list of candidate regions [start, end, sentiment_id, region_size]
    candidate_regions = torch.cat([
        sentiment_span_indices,
        sentiment_values,
        sentiment_span_indices.sum(dim=-1, keepdim=True)
    ], dim=-1).tolist()
    
    # Sort regions by size, then by start index
    candidate_regions.sort(key=lambda x: (x[-1], x[0]))

    # Step 4: Iterate through candidate regions to extract triplets
    valid_triplets = []
    valid_triplets_set = set()

    for start_idx, end_idx, sentiment_id, _ in candidate_regions:
        
        # CASE 1: Aspect-Opinion Order
        aspect_candidates = find_sub_spans(is_aspect_span[start_idx, start_idx:end_idx + 1], start_idx)
        opinion_candidates = find_sub_spans(is_opinion_span[start_idx:end_idx + 1, end_idx], start_idx)

        if aspect_candidates and opinion_candidates:
            # Select span based on confidence if available, otherwise use heuristic
            if confidence_scores is not None:
                aspect_choice = select_best_span_by_confidence(
                    aspect_candidates, end_idx, start_idx, 
                    confidence_scores.get('aspect'), 'row'
                )
                opinion_choice = select_best_span_by_confidence(
                    opinion_candidates, start_idx, end_idx,
                    confidence_scores.get('opinion'), 'col'
                )
            else:
                # Fallback to original heuristic
                aspect_choice = aspect_candidates[-1] if (len(aspect_candidates) == 1 or aspect_candidates[-1] != end_idx) else aspect_candidates[-2]
                opinion_choice = opinion_candidates[0] if (len(opinion_candidates) == 1 or opinion_candidates[0] != start_idx) else opinion_candidates[1]
            
            aspect_span = [start_idx, aspect_choice]
            opinion_span = [opinion_choice, end_idx]
            sentiment = id_to_sentiment[sentiment_id]
            
            triplet = (tuple(aspect_span), tuple(opinion_span), sentiment)
            if str(triplet) not in valid_triplets_set:
                valid_triplets.append(triplet)
                valid_triplets_set.add(str(triplet))

        # CASE 2: Opinion-Aspect Order
        opinion_candidates = find_sub_spans(is_opinion_span[start_idx, start_idx:end_idx + 1], start_idx)
        aspect_candidates = find_sub_spans(is_aspect_span[start_idx:end_idx + 1, end_idx], start_idx)

        if aspect_candidates and opinion_candidates:
            if confidence_scores is not None:
                opinion_choice = select_best_span_by_confidence(
                    opinion_candidates, end_idx, start_idx,
                    confidence_scores.get('opinion'), 'row'
                )
                aspect_choice = select_best_span_by_confidence(
                    aspect_candidates, start_idx, end_idx,
                    confidence_scores.get('aspect'), 'col'
                )
            else:
                # Fallback to original heuristic
                opinion_choice = opinion_candidates[-1] if (len(opinion_candidates) == 1 or opinion_candidates[-1] != end_idx) else opinion_candidates[-2]
                aspect_choice = aspect_candidates[0] if (len(aspect_candidates) == 1 or aspect_candidates[0] != start_idx) else aspect_candidates[1]
            
            opinion_span = [start_idx, opinion_choice]
            aspect_span = [aspect_choice, end_idx]
            sentiment = id_to_sentiment[sentiment_id]
            
            triplet = (tuple(aspect_span), tuple(opinion_span), sentiment)
            if str(triplet) not in valid_triplets_set:
                valid_triplets.append(triplet)
                valid_triplets_set.add(str(triplet))

    # Step 5: Return all found entities and the final sorted triplets
    return {
        'aspects': is_aspect_span.nonzero().squeeze().tolist(),
        'opinions': is_opinion_span.nonzero().squeeze().tolist(),
        'triplets': sorted(valid_triplets, key=lambda x: (x[0][0], x[0][-1], x[1][0], x[1][-1]))
    }

def select_best_span_by_confidence(candidates, fixed_idx, start_idx, conf_matrix, direction):
    """
    Select the best span from candidates based on confidence scores.
    
    Args:
        candidates (list): List of candidate span end/start indices
        fixed_idx (int): The fixed index (row or column)
        start_idx (int): Starting index for the span
        conf_matrix (tensor or None): Confidence matrix
        direction (str): 'row' or 'col' to indicate how to index confidence matrix
    
    Returns:
        int: The selected candidate index with highest confidence
    """
    if conf_matrix is None or len(candidates) == 1:
        # Fallback to boundary heuristic
        return candidates[-1] if candidates[-1] != fixed_idx else candidates[-2] if len(candidates) > 1 else candidates[0]
    
    conf_tensor = torch.tensor(conf_matrix) if not isinstance(conf_matrix, torch.Tensor) else conf_matrix
    
    # Calculate average confidence for each candidate span
    best_candidate = candidates[0]
    best_confidence = -1
    
    for candidate in candidates:
        if direction == 'row':
            # For aspect in row: confidence from start_idx to candidate
            span_conf = conf_tensor[start_idx, start_idx:candidate + 1].mean().item()
        else:  # 'col'
            # For opinion in column: confidence from candidate to fixed_idx
            span_conf = conf_tensor[candidate:fixed_idx + 1, fixed_idx].mean().item()
        
        if span_conf > best_confidence:
            best_confidence = span_conf
            best_candidate = candidate
    
    return best_candidate

def find_sub_spans(span_mask, offset):
    """Helper to find and return indices of true values in a boolean mask."""
    indices = (span_mask.nonzero().squeeze() + offset).tolist()
    return ensure_list(indices)

def ensure_list(item):
    """Ensures that the returned item is always a list."""
    if not isinstance(item, list):
        return [item]
    return item
