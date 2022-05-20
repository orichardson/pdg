---
title: PDG Viz Desires
---
<style>
	li > ul {
	  padding-bottom: 0.7em;
	}
 </style>

# Queue
  * import / export
  * Refactor drawing modes
  * Actions + Keystroke Display
  * visual
	  * draw labels
	  * change \delta (edge offset) + length parameters of hyperedges
	  * make self-edges prettier
  * Edge Focus
	  * cpd editor
  * Node focus
	  * value editor + display
  * UNDO
  * Multi-PDG
	  * Nested borders (Compress full PDGs as nodes + folding)
	  * Split and merge functionality
	  * Move entire PDGs around screen at once (nested SVG <g>'s)



## Functionality Wishlist

 - **Modeling Tool**
	 - Other Inputs (far future)
		 - Draw from Tablet
		 - Parse from image
	 - Import...
		 - .csv
		 - graph formats (.dot, ?)
		 - .json
		 - .pdg
		 - 

 - **Queries** 
	 - inconsistency
	 - joint distribution  
 
 - **Analytics**
	 - information diagrams (both for dists and pdgs)

	 
	 
# DONE

* [X]- bounding box for nodes
* [X]- edge labels in hypergraph
* [X]- new "node" for each hypergraph
  * [X]- repulsion between edges
* [X]- add "new node" tool
* [X]- add "new hyperedge" tool
* [X]- selection: (nodes + edges)
  * [X]- selection operators: union, subtraction
  * [X]- drawing selection
  * [X]- painting selection

* [X]- add to hyperedges w/drawing
* [X]- select edges
