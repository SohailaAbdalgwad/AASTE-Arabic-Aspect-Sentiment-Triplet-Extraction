import torch

def extract_triplets_from_tags(tag_table, id_to_sentiment, version='3D'):
    """
    Decodes a tag table from the model into a list of aspect-sentiment-opinion triplets.

    This function implements a greedy inference algorithm to find the most likely
    triplets based on the model's output grid.

    Args:
        tag_table (list of lists): The 2D grid of predicted tag IDs from the model.
        id_to_sentiment (dict): A mapping from sentiment ID to sentiment string (e.g., {1: 'POS'}).
        version (str, optional): The encoding version of the tags ('1D' or '3D'). Defaults to '3D'.

    Returns:
        dict: A dictionary containing lists of found 'aspects', 'opinions', and 'triplets'.
    """
    tag_table_tensor = torch.tensor(tag_table)

    # Step 1: Decode the bitmask tags into separate boolean masks for aspects, opinions, and sentiments.
    if version == '1D':  # {N, NEG, NEU, POS, O, A}
        is_aspect_span = (tag_table_tensor == 5) > 0
        is_opinion_span = (tag_table_tensor == 4) > 0
        sentiment_grid = tag_table_tensor * ((tag_table_tensor > 0) & (tag_table_tensor < 4))
    else:  # 3D: {N,A} - {N,O} - {N, NEG, NEU, POS}
        is_aspect_span = (tag_table_tensor & 8) > 0
        is_opinion_span = (tag_table_tensor & 4) > 0
        sentiment_grid = (tag_table_tensor & 3)

    # Step 2: Identify all spans that have a sentiment label. These are our candidate regions for triplets.
    sentiment_span_indices = sentiment_grid.nonzero()
    sentiment_values = sentiment_grid[sentiment_span_indices[:, 0], sentiment_span_indices[:, 1]].unsqueeze(dim=-1)
    
    # Create a list of candidate regions [start, end, sentiment_id, region_size]
    candidate_regions = torch.cat([
        sentiment_span_indices,
        sentiment_values,
        sentiment_span_indices.sum(dim=-1, keepdim=True)
    ], dim=-1).tolist()
    
    # Sort regions by size, then by start index. This greedy approach processes smaller spans first.
    candidate_regions.sort(key=lambda x: (x[-1], x[0]))

    # Step 3: Iterate through candidate regions to extract triplets.
    valid_triplets = []
    valid_triplets_set = set()

    for start_idx, end_idx, sentiment_id, _ in candidate_regions:
        
        # CASE 1: Aspect-Opinion Order
        # Find all aspect spans within the current region's row.
        aspect_candidates = find_sub_spans(is_aspect_span[start_idx, start_idx:end_idx + 1], start_idx)
        # Find all opinion spans within the current region's column.
        opinion_candidates = find_sub_spans(is_opinion_span[start_idx:end_idx + 1, end_idx], start_idx)

        if aspect_candidates and opinion_candidates:
            # This heuristic selects the most likely aspect/opinion from the candidates.
            # It prefers spans that are not at the boundaries of the region unless it's the only option.
            aspect_choice_idx = -1 if (len(aspect_candidates) == 1 or aspect_candidates[-1] != end_idx) else -2
            opinion_choice_idx = 0 if (len(opinion_candidates) == 1 or opinion_candidates[0] != start_idx) else 1
            
            aspect_span = [start_idx, aspect_candidates[aspect_choice_idx]]
            opinion_span = [opinion_candidates[opinion_choice_idx], end_idx]
            sentiment = id_to_sentiment[sentiment_id]
            
            triplet = (tuple(aspect_span), tuple(opinion_span), sentiment)
            if str(triplet) not in valid_triplets_set:
                valid_triplets.append(triplet)
                valid_triplets_set.add(str(triplet))

        # CASE 2: Opinion-Aspect Order
        # Find all opinion spans within the current region's row.
        opinion_candidates = find_sub_spans(is_opinion_span[start_idx, start_idx:end_idx + 1], start_idx)
        # Find all aspect spans within the current region's column.
        aspect_candidates = find_sub_spans(is_aspect_span[start_idx:end_idx + 1, end_idx], start_idx)

        if aspect_candidates and opinion_candidates:
            # Apply the same selection heuristic as in Case 1.
            opinion_choice_idx = -1 if (len(opinion_candidates) == 1 or opinion_candidates[-1] != end_idx) else -2
            aspect_choice_idx = 0 if (len(aspect_candidates) == 1 or aspect_candidates[0] != start_idx) else 1
            
            opinion_span = [start_idx, opinion_candidates[opinion_choice_idx]]
            aspect_span = [aspect_candidates[aspect_choice_idx], end_idx]
            sentiment = id_to_sentiment[sentiment_id]
            
            triplet = (tuple(aspect_span), tuple(opinion_span), sentiment)
            if str(triplet) not in valid_triplets_set:
                valid_triplets.append(triplet)
                valid_triplets_set.add(str(triplet))

    # Step 4: Return all found entities and the final sorted triplets.
    return {
        'aspects': is_aspect_span.nonzero().squeeze().tolist(),
        'opinions': is_opinion_span.nonzero().squeeze().tolist(),
        'triplets': sorted(valid_triplets, key=lambda x: (x[0][0], x[0][-1], x[1][0], x[1][-1]))
    }

def find_sub_spans(span_mask, offset):
    """Helper to find and return indices of true values in a boolean mask."""
    indices = (span_mask.nonzero().squeeze() + offset).tolist()
    return ensure_list(indices)

def ensure_list(item):
    """Ensures that the returned item is always a list."""
    if not isinstance(item, list):
        return [item]
    return item
