use core::f32;
use std::collections::{HashMap, HashSet};

use crate::heap::GeneralHeap;

// #[derive(Default)]
pub struct Graph {
    adjacency_list: HashMap<i32, GraphNode>,
    calculated_traversal: HashMap<i32, (f32, Vec<i32>)>,
}

impl Graph {
    pub fn new(adjacency_list: HashMap<i32, GraphNode>) -> Self {
        Self {
            adjacency_list,
            calculated_traversal: HashMap::new(),
        }
    }
    fn reset_costs_and_paths(&mut self,start_node:i32){
        for (key,value) in self.adjacency_list.iter_mut(){
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
            vec![Traversal_Node {
                node_key: start,
                cost: 0.0,
            }],
            |a| a.cost,
        );

        while let Some(current_node) = priority_queue.extract_min() {
            if current_node.node_key==finish{
                break;
            }
            let neighbours = match self.adjacency_list.get(&current_node.node_key) {
                Some(node) => node.neighbours.to_vec(),
                None => return None,
            };
            for neighbour in neighbours {
                let distance = current_node.cost + neighbour.node_edge_weight;
                let current_cost_of_neighbour=match self.adjacency_list.get(&neighbour.node_key){
                    Some(neighbour_info) =>neighbour_info.cost ,
                    None => return  None,
                };
                if distance < current_cost_of_neighbour {
                    if let Some(neighbour_ref) = self.adjacency_list.get_mut(&neighbour.node_key) {
                        neighbour_ref.cost = distance;
                        neighbour_ref.previous = Some(current_node.node_key);
                        priority_queue.insert(Traversal_Node { node_key: neighbour.node_key, cost: distance });
                    };
                }
            }
        }
        let mut path=Vec::new();
        path.push(finish);
        let mut current=finish;
        while let Some(current_node)=self.adjacency_list.get(&current){
            if let Some(previous)=current_node.previous{
                path.push(previous);
                current=previous;
            }else{
                break;
            }
        }
        let path_cost=match self.adjacency_list.get(&finish){
            Some(node) => node.cost,
            None => return  None,
        };
        Some(ShortestPathStats { path: path.into_iter().rev().collect(), cost:path_cost})
    }
}

#[derive(Clone,Debug)]
pub struct Neighbour {
    pub node_key: i32,
    pub node_edge_weight: f32,
}
#[derive(Debug)]
pub struct GraphNode {
    pub neighbours: Vec<Neighbour>,
    pub cost: f32,
    pub previous: Option<i32>,
}
#[derive(Debug)]
pub struct  ShortestPathStats{
    pub path:Vec<i32>,
    pub cost:f32
}

pub struct Traversal_Node {
    pub node_key: i32,
    pub cost: f32,
}
