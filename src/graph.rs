use core::f32;
use std::{collections::HashMap, error::Error, fmt::Display, hash::Hash};

use crate::heap::GeneralHeap;

// #[derive(Default)]
pub struct Graph <T>{
    adjacency_list: HashMap<T, Vec<Neighbour<T>>>,
    nodes_paths_and_costs:HashMap<T,NodePathCost<T>>
}

impl<T> Graph<T>
where T:Eq+Hash+Clone {
    pub fn new(adjacency_list: HashMap<T, Vec<Neighbour<T>>>) -> Self {
        let mut nodes_paths_and_costs=HashMap::new();
        for key in adjacency_list.keys(){
            nodes_paths_and_costs.insert(key.clone(), NodePathCost{
                previous:None,
                cost:f32::INFINITY
            });

        }
        Self {
            adjacency_list,
            nodes_paths_and_costs
        }
    }
    fn reset_costs_and_paths(&mut self,start_node:&T){
        for (key,value) in self.nodes_paths_and_costs.iter_mut(){
            value.previous=None;
            if key==start_node{
                value.cost=0.0
            }else{
                value.cost=f32::INFINITY;

            }
        }
    }
    pub fn shortest_path<'a>(&'a mut self, start: T, finish: &'a T) -> Option<ShortestPathStats<'a,T>> {
        self.reset_costs_and_paths(&start);
        let mut priority_queue = GeneralHeap::new(
            vec![TraversalNode {
                node_key: &start,
                cost: 0.0,
            }]
        );

        while let Some(current_node) = priority_queue.extract_min() {
            if current_node.node_key==finish{
                break;
            }
            let neighbours = self.adjacency_list.get(current_node.node_key)?;
            for neighbour in neighbours {
                let distance = current_node.cost + neighbour.node_edge_weight;
                let current_cost_of_neighbour=self.nodes_paths_and_costs.get(&neighbour.node_key)?.cost;
                if distance < current_cost_of_neighbour {
                    if let Some(neighbour_ref) = self.nodes_paths_and_costs.get_mut(&neighbour.node_key) {
                        neighbour_ref.cost = distance;
                        neighbour_ref.previous = Some(current_node.node_key.clone());
                        priority_queue.insert(TraversalNode { node_key: &neighbour.node_key, cost: distance });
                    };
                }
            }
        }
        let mut path=Vec::new();
        path.push(finish);
        let mut current=finish;
        while let Some(current_node)=self.nodes_paths_and_costs.get(current){
            if let Some(ref previous)=current_node.previous{
                path.push(previous);
                current=previous
            }else{
                break;
            }
        }
        let path_cost=self.nodes_paths_and_costs.get(finish)?.cost;
        Some(ShortestPathStats { path: path.into_iter().rev().collect(), cost:path_cost})
    }
}

#[derive(Debug)]
pub struct Neighbour<T> {
    node_key: T,
    node_edge_weight: f32,
}
impl<T> Neighbour<T>
where T:Eq+Hash+Clone {
    pub fn new(node_key:T,node_edge_weight:f32)->Neighbour<T>{
        Neighbour { node_key, node_edge_weight }

    }
}
pub struct NodePathCost<T>{
    cost: f32,
    previous: Option< T>,

}
impl<T> NodePathCost<T>
where T:Eq+Hash+Clone
{
    pub fn new(cost:f32)->NodePathCost<T>{
        NodePathCost{
            cost,
            previous:None
        }
    }
}
#[derive(Debug)]
pub struct  ShortestPathStats<'a,T>{
    path:Vec<&'a T>,
    cost:f32
}

struct TraversalNode<'a,T> {
    node_key: &'a T,
    cost: f32,
}
impl<'a,T> TraversalNode<'a,T>
where T:Eq+Hash+Clone {
    pub fn new<'b>(node_key:&'b T,cost:f32)->TraversalNode<'b,T>{
        TraversalNode { node_key, cost } 

    }
}
impl<'a,T> PartialEq for TraversalNode<'a,T> {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl<'a,T> PartialOrd for TraversalNode<'a,T>{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        
        self.cost.partial_cmp(&other.cost)
    }
}

#[derive(Debug)]
pub enum BuildingGraphFromMatrixError{
    EmptyMatrix,
    NotSquareMatrix
}
impl Display for BuildingGraphFromMatrixError{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self{
            BuildingGraphFromMatrixError::EmptyMatrix =>  write!(f, "the adjacency matrix is empty,provide a valid one!"),
            BuildingGraphFromMatrixError::NotSquareMatrix => write!(f, "the rows and columns of the adjacency matrix must be equal"),
        }
    }
}
impl Error for BuildingGraphFromMatrixError{}

impl TryFrom<Vec<Vec<f32>>> for Graph<usize>{
    type Error=BuildingGraphFromMatrixError;

    fn try_from(adjacency_matrix: Vec<Vec<f32>>) -> Result<Self, Self::Error> {
        if adjacency_matrix.is_empty(){
            return Err(BuildingGraphFromMatrixError::EmptyMatrix);
        }
        let rows=adjacency_matrix.len();
        let columns=adjacency_matrix[0].len();
        if rows !=columns{
            return Err(BuildingGraphFromMatrixError::NotSquareMatrix);

        }
        let mut adjacency_list=HashMap::<usize,Vec<Neighbour<usize>>>::new();
        for (index,row) in adjacency_matrix.iter().enumerate(){
            for (inner_index,edge_weight) in row.iter().enumerate(){
                if *edge_weight!=0.0{
                    let neighbours=adjacency_list.entry(index).or_default();
                    neighbours.push(Neighbour::new(inner_index, *edge_weight));

                }

            }

        }
        Ok(Graph::new(adjacency_list))
    }
}
