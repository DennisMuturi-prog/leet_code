use std::{collections::HashMap, str::Chars};

pub struct Node {
    val: usize,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}
impl Node {
    pub fn new(val: usize, left: Option<Box<Node>>, right: Option<Box<Node>>) -> Self {
        Self { val, left, right }
    }
}

pub struct Tree {
    root: Node,
}

impl Tree {
    pub fn new(root: Node) -> Self {
        Self { root }
    }
    pub fn find_path_from_root(&self, destination: usize) -> Vec<usize> {
        let mut traversal_list = vec![&self.root];
        let mut ready = vec![false];
        let mut path = Vec::new();

        while let Some(current) = traversal_list.pop() {
            let is_ready = ready.pop().unwrap();
            if is_ready {
                if current.val == destination {
                    return path;
                } else {
                    path.pop();
                }
            } else {
                path.push(current.val);
                traversal_list.push(current);
                ready.push(true);
                if let Some(ref right) = current.right {
                    traversal_list.push(right);
                    ready.push(false);
                }
                if let Some(ref left) = current.left {
                    traversal_list.push(left);
                    ready.push(false);
                }
            }
        }

        path
    }
    pub fn all_paths_from_root_to_leaf_nodes(&self) -> Vec<Vec<usize>> {
        let mut traversal_list = vec![&self.root];
        let mut ready = vec![false];
        let mut all_paths = Vec::new();
        let mut path = Vec::<usize>::new();

        while let Some(current) = traversal_list.pop() {
            let is_ready = ready.pop().unwrap();
            if is_ready {
                path.pop();
            } else {
                let mut is_leaf_node = true;
                path.push(current.val);
                traversal_list.push(current);
                ready.push(true);
                if let Some(ref right) = current.right {
                    is_leaf_node = false;
                    traversal_list.push(right);
                    ready.push(false);
                }
                if let Some(ref left) = current.left {
                    is_leaf_node = false;
                    traversal_list.push(left);
                    ready.push(false);
                }
                if is_leaf_node {
                    all_paths.push(path.clone());
                }
            }
        }

        all_paths
    }
}

#[derive(Default)]
pub struct TrieNode {
    is_end_of_word: bool,
    children: HashMap<char, Box<TrieNode>>,
}

#[derive(Default)]
pub struct Trie {
    root: TrieNode,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl Trie {
    pub fn insert(&mut self, word: String) {
        let mut current = &mut self.root;
        for letter in word.chars() {
            current = current
                .children
                .entry(letter)
                .or_insert(Box::new(TrieNode::default()));
        }
        current.is_end_of_word = true;
    }

    pub fn search(&self, word: &str) -> bool {
        let mut current = &self.root;
        for letter in word.chars() {
            current = match current.children.get(&letter) {
                Some(node) => node,
                None => {
                    return false;
                }
            }
        }
        current.is_end_of_word
    }

    pub fn starts_with(&self, prefix: String) -> bool {
        let mut current = &self.root;
        for letter in prefix.chars() {
            current = match current.children.get(&letter) {
                Some(node) => node,
                None => {
                    return false;
                }
            }
        }
        true
    }
    pub fn find_possible_matches(&mut self,word:String)->Vec<String>{
        if !self.starts_with(word.clone()){
            return Vec::new();
        }
        let mut current = &self.root;
        for letter in word.chars() {
            current = current.children.get(&letter).unwrap();
        }
        let mut path = word.chars().collect();
        let mut paths = Vec::new();
        let mut words = Vec::new();
        if current.is_end_of_word{
            words.push(word.clone());
        }
        for (key, value) in current.children.iter() {
            Trie::traverse_all_paths((*key, value), &mut path, &mut paths);
        }
        for word_path in paths {
            let word = String::from_iter(word_path);
            words.push(word);
        }
        words


    }
    pub fn find_all_words(&self) -> Vec<String> {
        let mut path = Vec::new();
        let mut paths = Vec::new();
        let mut words = Vec::new();
        for (key, value) in self.root.children.iter() {
            Trie::traverse_all_paths((*key, value), &mut path, &mut paths);
        }
        for word_path in paths {
            let word = String::from_iter(word_path);
            words.push(word);
        }
        words
    }
    pub fn traverse_all_paths(
        node: (char, &TrieNode),
        path: &mut Vec<char>,
        paths: &mut Vec<Vec<char>>,
    ) {
        path.push(node.0);
        if node.1.is_end_of_word {
            paths.push(path.clone());
        }
        for (key, value) in node.1.children.iter() {
            Trie::traverse_all_paths((*key, value), path, paths);
        }
        path.pop();
    }
    pub fn delete(&mut self, word: String) {
        if !self.search(&word) {
            return;
        }
        let mut chars = word.chars();
        Trie::delete_node(&mut self.root, &mut chars);
    }
    pub fn delete_node(node: &mut TrieNode, chars: &mut Chars) -> bool {
        match chars.next() {
            Some(current_char) => {
                let next_deletion =
                    Trie::delete_node(node.children.get_mut(&current_char).unwrap(), chars);
                if next_deletion {
                    node.children.remove(&current_char);
                }
                next_deletion && node.children.is_empty() && node.is_end_of_word
            }
            None => {
                node.is_end_of_word = false;
                node.children.is_empty()
            }
        }
    }
}
