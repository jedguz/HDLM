from hdlm.pipeline import GiddPipeline
import torch
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer

def semantic_kmeans_cluster(embeddings, cluster_size, max_iters=100, tolerance=1e-4, 
                           batch_size=1024, min_size_ratio=0.1, max_size_ratio=5.0, 
                           use_cosine=True, seed=42):
    """
    Semantically meaningful clustering with soft size constraints
    
    Args:
        embeddings: torch.Tensor [vocab_size, embed_dim] - word embeddings
        cluster_size: int - the number of clusters
        max_iters: int - the maximum number of iterations
        tolerance: float - the convergence threshold
        batch_size: int - the batch size, used for large vocabulary
        min_size_ratio: float - minimum cluster size as ratio of average size
        max_size_ratio: float - maximum cluster size as ratio of average size
        use_cosine: bool - whether to use cosine similarity instead of Euclidean distance
        seed: int - the random seed
    
    Returns:
        cluster_ids: torch.Tensor [vocab_size] - the cluster ids for each word
        centroids: torch.Tensor [cluster_size, embed_dim] - the cluster centers
    """
    torch.manual_seed(seed)
    device = embeddings.device
    vocab_size, embed_dim = embeddings.shape
    
    # Normalize embeddings if using cosine similarity
    if use_cosine:
        embeddings_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        normalized_embeddings = embeddings / embeddings_norm
    else:
        normalized_embeddings = embeddings
    
    # Calculate target cluster size bounds
    avg_cluster_size = vocab_size / cluster_size
    min_cluster_size = int(avg_cluster_size * min_size_ratio)
    max_cluster_size = int(avg_cluster_size * max_size_ratio)
    print(f"Target avg cluster size: {avg_cluster_size:.1f}, min: {min_cluster_size}, max: {max_cluster_size}")
    
    # 1. Initialize centroids with improved K-means++
    if use_cosine:
        centroids = torch.zeros(cluster_size, embed_dim, device=device)
        # Randomly select the first center
        first_centroid_idx = torch.randint(0, vocab_size, (1,))
        centroids[0] = normalized_embeddings[first_centroid_idx]
        centroids[0] /= torch.norm(centroids[0])
        
        # Select the remaining centers
        for k in range(1, cluster_size):
            # Compute similarities
            similarities = torch.mm(normalized_embeddings, centroids[:k].t())  # [vocab_size, k]
            max_similarities, _ = similarities.max(dim=1)  # [vocab_size]
            
            # Convert similarities to distances: higher similarity = lower distance
            distances = 1 - max_similarities
            
            # Select next centroid with probability proportional to distance
            probs = distances ** 2
            probs /= probs.sum()
            next_centroid_idx = torch.multinomial(probs, 1)
            centroids[k] = normalized_embeddings[next_centroid_idx]
            centroids[k] /= torch.norm(centroids[k])
    else:
        # Standard K-means++ initialization with Euclidean distance
        centroids = torch.zeros(cluster_size, embed_dim, device=device)
        first_centroid_idx = torch.randint(0, vocab_size, (1,))
        centroids[0] = normalized_embeddings[first_centroid_idx]
        
        for k in range(1, cluster_size):
            distances = torch.cdist(normalized_embeddings, centroids[:k])  # [vocab_size, k]
            min_distances = distances.min(dim=1)[0]  # [vocab_size]
            
            probs = min_distances ** 2
            probs /= probs.sum()
            next_centroid_idx = torch.multinomial(probs, 1)
            centroids[k] = normalized_embeddings[next_centroid_idx]
    
    # 2. Iterative optimization
    prev_centroids = centroids.clone()
    cluster_ids = torch.zeros(vocab_size, dtype=torch.long, device=device)
    
    for iteration in tqdm(range(max_iters), desc="Semantic clustering"):
        # 2.1 Assignment phase with size regularization
        # Calculate affinities/distances between all points and centroids
        if use_cosine:
            affinities = torch.zeros(vocab_size, cluster_size, device=device)
            for i in range(0, vocab_size, batch_size):
                batch_end = min(i + batch_size, vocab_size)
                batch_embeddings = normalized_embeddings[i:batch_end]
                batch_affinities = torch.mm(batch_embeddings, centroids.t())  # [batch_size, cluster_size]
                affinities[i:batch_end] = batch_affinities
            
            # Initialize with basic similarity assignments
            _, cluster_ids = affinities.max(dim=1)  # [vocab_size]
            
            # Get current cluster sizes
            cluster_sizes = torch.bincount(cluster_ids, minlength=cluster_size)
            
            # Apply soft size constraints through iterative reassignment
            num_reassignments = 0
            
            # Pass 1: Handle oversized clusters
            oversized_clusters = torch.where(cluster_sizes > max_cluster_size)[0]
            for cluster_idx in oversized_clusters:
                excess_count = cluster_sizes[cluster_idx] - max_cluster_size
                cluster_mask = (cluster_ids == cluster_idx)
                cluster_points = torch.where(cluster_mask)[0]
                
                # For each point in the oversized cluster, calculate its affinity to all other clusters
                cluster_affinities = affinities[cluster_points]
                affinity_to_current = cluster_affinities[:, cluster_idx].clone()
                
                # Set affinity to current cluster temporarily to -inf to find second best
                cluster_affinities[:, cluster_idx] = -float('inf')
                
                # Find second best cluster for each point
                second_best_affinities, second_best_clusters = cluster_affinities.max(dim=1)
                
                # Calculate the "loss" in affinity from reassignment
                affinity_loss = affinity_to_current - second_best_affinities
                
                # Reassign points with the smallest affinity loss
                _, indices_to_move = affinity_loss.topk(excess_count, largest=False)
                points_to_move = cluster_points[indices_to_move]
                
                # Update assignments and counts
                for point in points_to_move:
                    new_cluster = affinities[point].clone()
                    new_cluster[cluster_idx] = -float('inf')  # Exclude current cluster
                    new_cluster_idx = new_cluster.argmax().item()
                    
                    cluster_ids[point] = new_cluster_idx
                    num_reassignments += 1
            
            # Pass 2: Handle undersized clusters
            # Recompute cluster sizes after previous reassignments
            cluster_sizes = torch.bincount(cluster_ids, minlength=cluster_size)
            undersized_clusters = torch.where(cluster_sizes < min_cluster_size)[0]
            
            for cluster_idx in undersized_clusters:
                needed_count = min_cluster_size - cluster_sizes[cluster_idx]
                
                # Find points that should be in this cluster but are assigned elsewhere
                cluster_affinities = affinities[:, cluster_idx]
                current_clusters = cluster_ids
                
                # Create a binary mask for candidates: high affinity to this cluster but assigned elsewhere
                candidates_mask = (current_clusters != cluster_idx)
                
                if candidates_mask.sum() > 0:
                    candidate_affinities = cluster_affinities[candidates_mask]
                    candidate_indices = torch.where(candidates_mask)[0]
                    
                    # Sort by affinity
                    sorted_affinities, sorted_indices = candidate_affinities.sort(descending=True)
                    points_to_move = candidate_indices[sorted_indices[:needed_count]]
                    
                    # Update assignments
                    cluster_ids[points_to_move] = cluster_idx
                    num_reassignments += len(points_to_move)
            
            print(f"Iteration {iteration+1}: {num_reassignments} points reassigned")
            
        else:
            # Euclidean distance version with size regularization
            all_distances = torch.zeros(vocab_size, cluster_size, device=device)
            for i in range(0, vocab_size, batch_size):
                batch_end = min(i + batch_size, vocab_size)
                batch_embeddings = normalized_embeddings[i:batch_end]
                distances = torch.cdist(batch_embeddings, centroids)
                all_distances[i:batch_end] = distances
            
            # Initial assignment
            _, cluster_ids = all_distances.min(dim=1)
            
            # Apply similar size regularization as in cosine version
            cluster_sizes = torch.bincount(cluster_ids, minlength=cluster_size)
            num_reassignments = 0
            
            # Handle oversized clusters
            oversized_clusters = torch.where(cluster_sizes > max_cluster_size)[0]
            for cluster_idx in oversized_clusters:
                excess_count = cluster_sizes[cluster_idx] - max_cluster_size
                cluster_mask = (cluster_ids == cluster_idx)
                cluster_points = torch.where(cluster_mask)[0]
                
                # Calculate distance differences
                cluster_distances = all_distances[cluster_points]
                distance_to_current = cluster_distances[:, cluster_idx].clone()
                
                # Set distance to current cluster temporarily to inf to find second best
                cluster_distances[:, cluster_idx] = float('inf')
                
                # Find second best cluster for each point
                second_best_distances, second_best_clusters = cluster_distances.min(dim=1)
                
                # Calculate the "loss" in distance from reassignment
                distance_increase = second_best_distances - distance_to_current
                
                # Reassign points with the smallest distance increase
                _, indices_to_move = distance_increase.topk(excess_count, largest=False)
                points_to_move = cluster_points[indices_to_move]
                
                # Update assignments
                for point in points_to_move:
                    new_cluster = all_distances[point].clone()
                    new_cluster[cluster_idx] = float('inf')  # Exclude current cluster
                    new_cluster_idx = new_cluster.argmin().item()
                    
                    cluster_ids[point] = new_cluster_idx
                    num_reassignments += 1
            
            # Handle undersized clusters
            cluster_sizes = torch.bincount(cluster_ids, minlength=cluster_size)
            undersized_clusters = torch.where(cluster_sizes < min_cluster_size)[0]
            
            for cluster_idx in undersized_clusters:
                needed_count = min_cluster_size - cluster_sizes[cluster_idx]
                
                # Find points that should be in this cluster but are assigned elsewhere
                cluster_distances = all_distances[:, cluster_idx]
                current_clusters = cluster_ids
                
                # Create a binary mask for candidates
                candidates_mask = (current_clusters != cluster_idx)
                
                if candidates_mask.sum() > 0:
                    candidate_distances = cluster_distances[candidates_mask]
                    candidate_indices = torch.where(candidates_mask)[0]
                    
                    # Sort by distance (smaller is better)
                    sorted_distances, sorted_indices = candidate_distances.sort()
                    points_to_move = candidate_indices[sorted_indices[:needed_count]]
                    
                    # Update assignments
                    cluster_ids[points_to_move] = cluster_idx
                    num_reassignments += len(points_to_move)
            
            print(f"Iteration {iteration+1}: {num_reassignments} points reassigned")
        
        # 2.2 Update phase
        new_centroids = torch.zeros_like(centroids)
        for k in range(cluster_size):
            cluster_mask = (cluster_ids == k)
            if cluster_mask.sum() > 0:
                if use_cosine:
                    # For cosine similarity, normalize the mean vector
                    mean_vector = normalized_embeddings[cluster_mask].mean(dim=0)
                    new_centroids[k] = mean_vector / torch.norm(mean_vector)
                else:
                    new_centroids[k] = normalized_embeddings[cluster_mask].mean(dim=0)
        
        # 2.3 Check convergence
        if use_cosine:
            # For cosine similarity, measure angular change
            cos_similarities = torch.sum(centroids * new_centroids, dim=1)
            centroid_change = 1.0 - cos_similarities.mean().item()
        else:
            centroid_change = torch.norm(new_centroids - centroids).item()
        
        centroids = new_centroids
        
        print(f"Iteration {iteration+1}: centroid change = {centroid_change:.6f}")
        if centroid_change < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break
            
        prev_centroids = centroids.clone()
    
    # 3. Final cluster statistics
    final_cluster_sizes = torch.bincount(cluster_ids, minlength=cluster_size)
    print("Final cluster sizes:", final_cluster_sizes.tolist())
    
    # For semantic coherence, calculate average pairwise similarity within clusters
    coherence_scores = []
    for k in range(cluster_size):
        cluster_mask = (cluster_ids == k)
        if cluster_mask.sum() > 1:  # Need at least 2 points for pairwise similarity
            cluster_points = normalized_embeddings[cluster_mask]
            # Sample at most 1000 points to avoid memory issues with large clusters
            if cluster_mask.sum() > 1000:
                indices = torch.randperm(cluster_mask.sum())[:1000]
                cluster_points = cluster_points[indices]
            
            # Calculate pairwise similarities
            if use_cosine:
                similarity_matrix = torch.mm(cluster_points, cluster_points.t())
                # Exclude self-similarities
                similarity_sum = similarity_matrix.sum() - similarity_matrix.trace()
                n_pairs = cluster_points.size(0) * (cluster_points.size(0) - 1)
                avg_similarity = similarity_sum / n_pairs
            else:
                distance_matrix = torch.cdist(cluster_points, cluster_points)
                # Exclude self-distances (which are 0)
                distance_sum = distance_matrix.sum()
                n_pairs = cluster_points.size(0) * (cluster_points.size(0) - 1)
                avg_distance = distance_sum / n_pairs
                avg_similarity = 1.0 / (1.0 + avg_distance)  # Convert distance to similarity
            
            coherence_scores.append(avg_similarity.item())
        else:
            coherence_scores.append(0.0)
    
    print("Cluster coherence scores:", [f"{score:.4f}" for score in coherence_scores])
    overall_coherence = sum(coherence_scores) / len(coherence_scores)
    print(f"Overall semantic coherence: {overall_coherence:.4f}")
    
    return cluster_ids, centroids


def evaluate_clustering(embeddings, cluster_ids, centroids, use_cosine=True):
    """
    Evaluate the clustering results
    
    Args:
        embeddings: torch.Tensor [vocab_size, embed_dim] - the word embeddings
        cluster_ids: torch.Tensor [vocab_size] - the cluster ids
        centroids: torch.Tensor [cluster_size, embed_dim] - the cluster centers
        use_cosine: bool - whether to use cosine similarity
    """
    vocab_size = len(cluster_ids)
    cluster_size = len(centroids)
    
    # Normalize embeddings if using cosine
    if use_cosine:
        embeddings_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        normalized_embeddings = embeddings / embeddings_norm
        centroids_norm = torch.norm(centroids, p=2, dim=1, keepdim=True)
        normalized_centroids = centroids / centroids_norm
    else:
        normalized_embeddings = embeddings
        normalized_centroids = centroids
    
    # 1. Calculate the silhouette coefficient
    sample_size = min(1000, vocab_size)
    indices = torch.randperm(vocab_size)[:sample_size]
    
    sample_embeddings = normalized_embeddings[indices]
    sample_cluster_ids = cluster_ids[indices]
    
    # Calculate distances/similarities between samples
    if use_cosine:
        # For cosine, compute similarity matrix, then convert to distance
        similarities = torch.mm(sample_embeddings, sample_embeddings.t())
        distances = 1 - similarities  # Convert similarity to distance
    else:
        distances = torch.cdist(sample_embeddings, sample_embeddings)
    
    a = torch.zeros(sample_size, device=embeddings.device)
    b = torch.zeros(sample_size, device=embeddings.device)
    
    for i in range(sample_size):
        # Calculate a(i): the average distance to other points in the same cluster
        same_cluster = (sample_cluster_ids == sample_cluster_ids[i])
        same_cluster[i] = False  # Exclude itself
        if same_cluster.sum() > 0:
            a[i] = distances[i][same_cluster].mean()
        
        # Calculate b(i): the minimum average distance to other clusters
        b_candidates = torch.zeros(cluster_size, device=embeddings.device)
        for k in range(cluster_size):
            if k != sample_cluster_ids[i]:
                other_cluster = (sample_cluster_ids == k)
                if other_cluster.sum() > 0:
                    b_candidates[k] = distances[i][other_cluster].mean()
        
        # Only consider clusters that actually have points
        valid_b = b_candidates[b_candidates > 0]
        if len(valid_b) > 0:
            b[i] = valid_b.min()
    
    silhouette = (b - a) / torch.maximum(a, b)
    avg_silhouette = silhouette.mean().item()
    
    print(f"Average Silhouette Coefficient: {avg_silhouette:.4f}")
    
    # 2. Calculate semantic coherence within clusters
    cluster_coherence = torch.zeros(cluster_size, device=embeddings.device)
    cluster_sizes = torch.bincount(cluster_ids, minlength=cluster_size)
    
    for k in range(cluster_size):
        cluster_mask = (cluster_ids == k)
        if cluster_mask.sum() > 0:
            cluster_embeddings = normalized_embeddings[cluster_mask]
            
            if use_cosine:
                # Calculate similarity to centroid
                similarities = torch.mm(cluster_embeddings, normalized_centroids[k].unsqueeze(1)).squeeze()
                cluster_coherence[k] = similarities.mean()
            else:
                # Calculate distance to centroid
                distances = torch.norm(cluster_embeddings - normalized_centroids[k], dim=1)
                cluster_coherence[k] = 1.0 / (1.0 + distances.mean())  # Convert to similarity-like measure
    
    print("\nCluster statistics:")
    for k in range(cluster_size):
        print(f"Cluster {k}: size={cluster_sizes[k]}, coherence={cluster_coherence[k]:.4f}")
    
    return {
        'silhouette': avg_silhouette,
        'cluster_sizes': cluster_sizes,
        'cluster_coherence': cluster_coherence,
        'overall_coherence': cluster_coherence.mean().item()
    }


# Keep the visualization functions as they are
def visualize_clusters(embeddings, cluster_ids, tokenizer=None, sample_size=2000, perplexity=30, seed=42):
    """
    visualize the clustering results of the word embeddings using t-SNE
    
    Args:
        embeddings: torch.Tensor [vocab_size, embed_dim] - the word embeddings
        cluster_ids: torch.Tensor [vocab_size] - the cluster labels
        tokenizer: tokenizer (optional)
        sample_size: int - the sample size, used to handle large vocabulary
        perplexity: float - the perplexity parameter of t-SNE
        seed: int - the random seed
    """
    # 1. data preprocessing
    vocab_size = len(embeddings)
    num_clusters = cluster_ids.max().item() + 1
    
    # if the vocabulary is too large, randomly sample
    if vocab_size > sample_size:
        np.random.seed(seed)
        indices = np.random.choice(vocab_size, sample_size, replace=False)
        embeddings_sample = embeddings[indices].cpu().numpy()
        cluster_ids_sample = cluster_ids[indices].cpu().numpy()
        if tokenizer:
            tokens_sample = [tokenizer.decode([i]) for i in indices]
    else:
        embeddings_sample = embeddings.cpu().numpy()
        cluster_ids_sample = cluster_ids.cpu().numpy()
        if tokenizer:
            tokens_sample = [tokenizer.decode([i]) for i in range(vocab_size)]
    
    # 2. use t-SNE to reduce the dimension
    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init='pca',  # use PCA to initialize for better stability
        n_iter=1000
    )
    embeddings_2d = tsne.fit_transform(embeddings_sample)
    
    # 3. create the visualization
    plt.figure(figsize=(15, 15))
    
    # set the color scheme
    colors = sns.color_palette("husl", n_colors=num_clusters)
    
    # plot the scatter plot
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=[colors[i] for i in cluster_ids_sample],
        alpha=0.6,
        s=50
    )
    
    # add title and labels
    plt.title("t-SNE Visualization of Word Embeddings Clusters", fontsize=16)
    plt.xlabel("t-SNE dimension 1", fontsize=12)
    plt.ylabel("t-SNE dimension 2", fontsize=12)
    
    # add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[i], 
                                 label=f'Cluster {i}',
                                 markersize=10)
                      for i in range(num_clusters)]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 4. add interactive function (optional)
    if tokenizer:
        def on_click(event):
            if event.inaxes == plt.gca():
                # find the nearest point
                dist = np.sqrt((embeddings_2d[:, 0] - event.xdata)**2 + 
                             (embeddings_2d[:, 1] - event.ydata)**2)
                idx = dist.argmin()
                
                # show the word and the cluster information
                token = tokens_sample[idx]
                cluster = cluster_ids_sample[idx]
                plt.title(f"Word: {token}, Cluster: {cluster}")
                plt.draw()
        
        plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    
    # 5. save and show
    plt.tight_layout()
    plt.savefig("embedding_clusters.png", dpi=300, bbox_inches='tight')
    plt.show()

def visualize_cluster_statistics(cluster_ids, tokenizer=None, top_k=100):
    """
    visualize the statistics of the clustering results
    
    Args:
        cluster_ids: torch.Tensor [vocab_size] - the cluster labels
        tokenizer: tokenizer (optional)
        top_k: int - show the top k most common words from each cluster
    """
    num_clusters = cluster_ids.max().item() + 1
    cluster_sizes = torch.bincount(cluster_ids, minlength=num_clusters)
    
    # 1. plot the distribution of cluster sizes
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(cluster_sizes.cpu().numpy(), bins=30)
    plt.title("Distribution of Cluster Sizes")
    plt.xlabel("Cluster Size")
    plt.ylabel("Count")
    
    plt.subplot(1, 2, 2)
    plt.boxplot(cluster_sizes.cpu().numpy())
    plt.title("Cluster Size Box Plot")
    plt.ylabel("Cluster Size")
    
    plt.tight_layout()
    plt.show()
    
    # 2. show example words from each cluster
    if tokenizer:
        print("\nExample words from each cluster:")
        for cluster in range(num_clusters):
            cluster_tokens = torch.where(cluster_ids == cluster)[0]
            if len(cluster_tokens) > 0:
                example_tokens = cluster_tokens[500:700]  # min(top_k, len(cluster_tokens))
                example_words = [tokenizer.decode([i.item()]) for i in example_tokens]
                cluster_info = f"\nCluster {cluster} (size: {len(cluster_tokens)}):\n"
                words_info = ", ".join(example_words)
                
                # Print to console
                print(cluster_info.strip())
                print(words_info)
                
                # Write to file
                with open("cluster_examples.txt", "w", encoding="utf-8") as f:
                    f.write(cluster_info)
                    f.write(words_info + "\n")
                # example_tokens = cluster_tokens[:min(top_k, len(cluster_tokens))]
                # example_words = [tokenizer.decode([i.item()]) for i in example_tokens]
                # print(f"\nCluster {cluster} (size: {len(cluster_tokens)}):")
                # print(", ".join(example_words))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--cluster_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_size_ratio", type=float, default=0.1)
    parser.add_argument("--max_size_ratio", type=float, default=5.0)
    parser.add_argument("--use_cosine", action="store_true", help="Use cosine similarity instead of Euclidean distance")
    parser.add_argument("--dataset_name", type=str, default="owt")  # [owt, lm1b]
    args = parser.parse_args()

    model_name = args.model_name
    cluster_size = args.cluster_size

    # Download a pretrained model from HuggingFace
    # pipe = GiddPipeline.from_pretrained(model_name, trust_remote_code=True)

    # embeddings = pipe.model.vocab_embed.embedding.detach()
    model = AutoModel.from_pretrained(model_name)
    embeddings = model.embeddings.word_embeddings.weight.detach()
    # tokenizer = pipe.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)
    print(vocab_size, embeddings.shape)

    # Use the new semantic clustering approach
    cluster_dict, centroids = semantic_kmeans_cluster(
        embeddings=embeddings,
        cluster_size=cluster_size,
        max_iters=args.max_iters,
        tolerance=args.tolerance,
        batch_size=args.batch_size,
        min_size_ratio=args.min_size_ratio,
        max_size_ratio=args.max_size_ratio,
        use_cosine=args.use_cosine,
        seed=args.seed
    )

    dataset_tag = "" if args.dataset_name == "owt" else f"_{args.dataset_name}"
    torch.save(cluster_dict, f"semantic_cluster_dict_{model_name}_{cluster_size}{dataset_tag}.pt")  # visualize
    torch.save(centroids, f"semantic_centroids_{model_name}_{cluster_size}{dataset_tag}.pt")

    # Evaluate with updated metrics
    # metrics = evaluate_clustering(embeddings, cluster_dict, centroids, use_cosine=args.use_cosine)
    # print(metrics)

    # visualize_clusters(embeddings, cluster_dict, tokenizer=tokenizer, sample_size=1000, perplexity=30, seed=42)
    # visualize_cluster_statistics(cluster_dict, tokenizer=tokenizer, top_k=200)