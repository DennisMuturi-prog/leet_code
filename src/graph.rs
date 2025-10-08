use core::f32;
use std::collections::{HashMap};

use crate::heap::GeneralHeap;

// #[derive(Default)]
pub struct Graph {
    adjacency_list: HashMap<i32, Vec<Neighbour>>,
    nodes_paths_and_costs:HashMap<i32,NodePathCost>
}

impl Graph {
    pub fn new(adjacency_list: HashMap<i32, Vec<Neighbour>>) -> Self {
        let mut nodes_paths_and_costs=HashMap::new();
        for key in adjacency_list.keys(){
            nodes_paths_and_costs.insert(*key, NodePathCost{
                previous:None,
                cost:f32::INFINITY
            });

        }
        Self {
            adjacency_list,
            nodes_paths_and_costs
        }
    }
    fn reset_costs_and_paths(&mut self,start_node:i32){
        for (key,value) in self.nodes_paths_and_costs.iter_mut(){
            value.previous=None;
            if *key==start_node{
                value.cost=0.0
            }else{
                value.cost=f32::INFINITY;

            }
        }
    }
    pub fn shortest_path(&mut self, start: i32, finish: i32) -> Option<ShortestPathStats> {
        self.reset_costs_and_paths(start);
        let mut priority_queue = GeneralHeap::new(
            vec![TraversalNode {
                node_key: start,
                cost: 0.0,
            }],
            |a| a.cost,
        );

        while let Some(current_node) = priority_queue.extract_min() {
            if current_node.node_key==finish{
                break;
            }
            let neighbours = self.adjacency_list.get(&current_node.node_key)?;
            for neighbour in neighbours {
                let distance = current_node.cost + neighbour.node_edge_weight;
                let current_cost_of_neighbour=self.nodes_paths_and_costs.get(&neighbour.node_key)?.cost;
                if distance < current_cost_of_neighbour {
                    if let Some(neighbour_ref) = self.nodes_paths_and_costs.get_mut(&neighbour.node_key) {
                        neighbour_ref.cost = distance;
                        neighbour_ref.previous = Some(current_node.node_key);
                        priority_queue.insert(TraversalNode { node_key: neighbour.node_key, cost: distance });
                    };
                }
            }
        }
        let mut path=Vec::new();
        path.push(finish);
        let mut current=finish;
        while let Some(current_node)=self.nodes_paths_and_costs.get(&current){
            if let Some(previous)=current_node.previous{
                path.push(previous);
                current=previous;
            }else{
                break;
            }
        }
        let path_cost=self.nodes_paths_and_costs.get(&finish)?.cost;
        Some(ShortestPathStats { path: path.into_iter().rev().collect(), cost:path_cost})
    }
}

#[derive(Debug)]
pub struct Neighbour {
    pub node_key: i32,
    pub node_edge_weight: f32,
}
pub struct NodePathCost{
    pub cost: f32,
    pub previous: Option<i32>,

}
#[derive(Debug)]
pub struct  ShortestPathStats{
    pub path:Vec<i32>,
    pub cost:f32
}

pub struct TraversalNode {
    pub node_key: i32,
    pub cost: f32,
}
