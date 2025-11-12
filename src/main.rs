use central_tendencies::graph::{Graph, Neighbour};
use central_tendencies::tree::{Node, Tree};
use central_tendencies::union_find::UnionFind;
use std::collections::btree_set::Union;
use std::collections::{BTreeMap, HashMap, hash_set};

fn main() {
    // let adjacency_list = HashMap::from([
    //     (
    //         'A', // Node A (was 0)
    //         vec![
    //             Neighbour::new('C', 2.4), // Was node 2
    //             Neighbour::new('D', 2.2), // Was node 3
    //         ],
    //     ),
    //     (
    //         'B', // Node B (was 1)
    //         vec![
    //             Neighbour::new('D', 2.2), // Was node 3
    //             Neighbour::new('E', 2.5), // Was node 4
    //         ],
    //     ),
    //     (
    //         'C', // Node C (was 2)
    //         vec![
    //             Neighbour::new('A', 2.4), // Was node 0
    //             Neighbour::new('F', 2.0), // Was node 5
    //             Neighbour::new('G', 2.8), // Was node 6
    //         ],
    //     ),
    //     (
    //         'D', // Node D (was 3)
    //         vec![
    //             Neighbour::new('A', 2.2), // Was node 0
    //             Neighbour::new('B', 2.2), // Was node 1
    //             Neighbour::new('H', 3.1), // Was node 7
    //             Neighbour::new('G', 2.6), // Was node 6
    //             Neighbour::new('F', 3.4), // Was node 5
    //         ],
    //     ),
    //     (
    //         'E', // Node E (was 4)
    //         vec![
    //             Neighbour::new('B', 2.5), // Was node 1
    //             Neighbour::new('H', 2.1), // Was node 7
    //             Neighbour::new('G', 2.9), // Was node 6
    //         ],
    //     ),
    //     (
    //         'F', // Node F (was 5)
    //         vec![
    //             Neighbour::new('C', 2.0), // Was node 2
    //             Neighbour::new('D', 3.4), // Was node 3
    //             Neighbour::new('I', 2.8), // Was node 8
    //             Neighbour::new('J', 4.0), // Was node 9
    //         ],
    //     ),
    //     (
    //         'G', // Node G (was 6)
    //         vec![
    //             Neighbour::new('D', 2.6), // Was node 3
    //             Neighbour::new('C', 2.8), // Was node 2
    //             Neighbour::new('E', 2.9), // Was node 4
    //             Neighbour::new('I', 2.8), // Was node 8
    //             Neighbour::new('J', 2.4), // Was node 9
    //         ],
    //     ),
    //     (
    //         'H', // Node H (was 7)
    //         vec![
    //             Neighbour::new('D', 3.1), // Was node 3
    //             Neighbour::new('E', 2.1), // Was node 4
    //             Neighbour::new('I', 4.4), // Was node 8
    //             Neighbour::new('J', 2.6), // Was node 9
    //         ],
    //     ),
    //     (
    //         'I', // Node I (was 8)
    //         vec![
    //             Neighbour::new('F', 2.8), // Was node 5
    //             Neighbour::new('G', 2.8), // Was node 6
    //             Neighbour::new('H', 4.4), // Was node 7
    //         ],
    //     ),
    //     (
    //         'J', // Node J (was 9)
    //         vec![
    //             Neighbour::new('G', 2.4), // Was node 6
    //             Neighbour::new('H', 2.6), // Was node 7
    //             Neighbour::new('F', 4.0), // Was node 5
    //         ],
    //     ),
    // ]);

    // let mut graph = Graph::new(adjacency_list);
    // let result = graph.shortest_path('A', &'I');
    // println!("result is {:?}", result);

    // let adjacency_matrix = vec![
    //     vec![0.0, 0.0, 2.4, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     vec![0.0, 0.0, 0.0, 2.2, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     vec![2.4, 0.0, 0.0, 0.0, 0.0, 2.0, 2.8, 0.0, 0.0, 0.0],
    //     vec![2.2, 2.2, 0.0, 0.0, 0.0, 3.4, 2.6, 3.1, 0.0, 0.0],
    //     vec![0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 2.9, 2.1, 0.0, 0.0],
    //     vec![0.0, 0.0, 2.0, 3.4, 0.0, 0.0, 0.0, 0.0, 2.8, 4.0],
    //     vec![0.0, 0.0, 2.8, 2.6, 2.9, 0.0, 0.0, 0.0, 2.8, 2.4],
    //     vec![0.0, 0.0, 0.0, 3.1, 2.1, 0.0, 0.0, 0.0, 4.4, 2.6],
    //     vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.8, 2.8, 4.4, 0.0, 0.0],
    //     vec![0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.4, 2.6, 0.0, 0.0],
    // ];

    // let mut graph = Graph::try_from(adjacency_matrix).unwrap();
    // let result = graph.shortest_path(0, &8);
    // println!("result is {:?}", result);
    // println!("nth fibonnaci is {}",nth_fibonnaci(93));
    // let mut calculated_results=HashMap::new();
    // println!("nth fibonnaci is {}",fibonnaci_recursive(93,&mut calculated_results));

    // Create a sample binary tree:
    //         10
    //        /  \
    //       5    15
    //      / \   / \
    //     3   7 12  20
    //    /
    //   1

    // Build from bottom up
    // let node1 = Node::new(1, None, None);
    // let node3 = Node::new(3, Some(Box::new(node1)), None);
    // let node7 = Node::new(7, None, None);
    // let node5 = Node::new(5, Some(Box::new(node3)), Some(Box::new(node7)));

    // let node12 = Node::new(12, None, None);
    // let node20 = Node::new(20, None, None);
    // let node15 = Node::new(15, Some(Box::new(node12)), Some(Box::new(node20)));

    // let root = Node::new(10, Some(Box::new(node5)), Some(Box::new(node15)));

    // let tree = Tree::new(root);
    // let path = tree.find_path_from_root(7);
    // let all_paths = tree.all_paths_from_root_to_leaf_nodes();

    // println!("path is {:?}", path);
    // println!("all_paths is {:?}", all_paths);

    let a = "-23";
    let num: i32 = a.parse().unwrap();
    println!("num is {}", num);
}
//Definition for singly-linked list.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}
struct Solution;
impl Solution {
    pub fn merge_two_lists(
        list1: Option<Box<ListNode>>,
        list2: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        let mut list1 = list1;
        let mut list2 = list2;
        let mut dummy_node = None;
        let mut current = &mut dummy_node;

        loop {
            let mut head_list1 = match list1.take() {
                Some(head) => head,
                None => {
                    *current = list2;
                    break;
                }
            };
            let mut head_list2 = match list2.take() {
                Some(head) => head,
                None => {
                    *current = Some(head_list1);
                    break;
                }
            };
            if head_list1.val < head_list2.val {
                list1 = head_list1.next.take();
                list2 = Some(head_list2);
                *current = Some(head_list1);
            } else {
                list2 = head_list2.next.take();
                list1 = Some(head_list1);
                *current = Some(head_list2);
            }
            current = &mut current.as_mut().unwrap().next;
        }
        dummy_node
    }
}

// Definition for a binary tree node.
#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}
use core::{f32, num, time};
use std::cell::RefCell;
use std::cmp::{max, min};
use std::collections::{HashSet, VecDeque};
use std::os::windows::raw::SOCKET;
use std::rc::Rc;
impl Solution {
    pub fn invert_tree(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        root.inspect(|tree_root| {
            let left = Solution::invert_tree(tree_root.borrow_mut().left.take());
            let right = Solution::invert_tree(tree_root.borrow_mut().right.take());
            tree_root.borrow_mut().left = right;
            tree_root.borrow_mut().right = left;
        })
    }
}

impl Solution {
    pub fn flood_fill_bfs(image: Vec<Vec<i32>>, sr: i32, sc: i32, color: i32) -> Vec<Vec<i32>> {
        let mut image = image;
        let row_index = sr as usize;
        let column_index = sc as usize;

        let original_color = image[row_index][column_index];
        if color == original_color {
            return image;
        }
        image[row_index][column_index] = color;
        let rows = image.len();
        let columns = image[0].len();
        let mut neighbors = VecDeque::new();
        Solution::add_neighbors_positions(&mut neighbors, row_index, column_index, rows, columns);
        while let Some((first_row_index, first_column_index)) = neighbors.pop_front() {
            if image[first_row_index][first_column_index] != original_color {
                continue;
            }
            image[first_row_index][first_column_index] = color;
            Solution::add_neighbors_positions(
                &mut neighbors,
                first_row_index,
                first_column_index,
                rows,
                columns,
            );
        }
        image
    }

    fn add_neighbors_positions(
        neighbors: &mut VecDeque<(usize, usize)>,
        row_index: usize,
        column_index: usize,
        rows: usize,
        columns: usize,
    ) {
        if let Some(new_column_index) = column_index.checked_sub(1) {
            if new_column_index < columns {
                neighbors.push_back((row_index, new_column_index));
            }
        }
        let new_column_index = column_index + 1;
        if new_column_index < columns {
            neighbors.push_back((row_index, new_column_index));
        }
        if let Some(new_row_index) = row_index.checked_sub(1) {
            if new_row_index < rows {
                neighbors.push_back((new_row_index, column_index));
            }
        }
        let new_row_index = row_index + 1;
        if new_row_index < rows {
            neighbors.push_back((new_row_index, column_index));
        }
    }
    pub fn bfs() {}
    pub fn flood_fill(image: Vec<Vec<i32>>, sr: i32, sc: i32, color: i32) -> Vec<Vec<i32>> {
        let mut image = image;
        let row_index = sr as usize;
        let column_index = sc as usize;

        let original_color = image[row_index][column_index];
        if color == original_color {
            return image;
        }
        let rows = image.len();
        let columns = image[0].len();
        Solution::dfs(
            &mut image,
            row_index,
            column_index,
            original_color,
            color,
            rows,
            columns,
        );
        image
    }
    fn dfs(
        image: &mut Vec<Vec<i32>>,
        row_index: usize,
        column_index: usize,
        original_color: i32,
        new_color: i32,
        rows: usize,
        columns: usize,
    ) {
        if image[row_index][column_index] != original_color {
            return;
        }
        image[row_index][column_index] = new_color;

        let mut combinations = Vec::new();
        if let Some(new_column_index) = column_index.checked_sub(1) {
            if new_column_index < columns {
                combinations.push((row_index, new_column_index));
            }
        }
        let new_column_index = column_index + 1;
        if new_column_index < columns {
            combinations.push((row_index, new_column_index));
        }
        if let Some(new_row_index) = row_index.checked_sub(1) {
            if new_row_index < rows {
                combinations.push((new_row_index, column_index));
            }
        }
        let new_row_index = row_index + 1;
        if new_row_index < rows {
            combinations.push((new_row_index, column_index));
        }
        for combination in combinations {
            Solution::dfs(
                image,
                combination.0,
                combination.1,
                original_color,
                new_color,
                rows,
                columns,
            );
        }
    }
}

impl Solution {
    pub fn lowest_common_ancestor_binary_search_tree(
        root: Option<Rc<RefCell<TreeNode>>>,
        p: Option<Rc<RefCell<TreeNode>>>,
        q: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        let p_node = p.unwrap();
        let p_value = p_node.borrow().val;
        let q_node = q.unwrap();
        let q_value = q_node.borrow().val;
        let mut current_node = root;
        while let Some(node) = current_node {
            let node_value = node.borrow().val;
            if p_value < node_value && q_value < node_value {
                current_node = node.borrow_mut().left.take();
            } else if p_value > node_value && q_value > node_value {
                current_node = node.borrow_mut().right.take();
            } else {
                return Some(node);
            }
        }
        None
    }
}

impl Solution {
    pub fn is_balanced_2(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        let mut nodes = VecDeque::new();
        nodes.push_back(root);
        while let Some(root_of_tree) = nodes.pop_front() {
            if let Some(current_root) = root_of_tree {
                let left_height = Solution::height_2(&current_root.borrow().left, 0);
                let right_height = Solution::height_2(&current_root.borrow().right, 0);
                let difference = (left_height as i32 - right_height as i32).abs();
                if difference > 1 {
                    return false;
                }
                let left_child = current_root.borrow_mut().left.take();
                let right_child = current_root.borrow_mut().right.take();
                nodes.push_back(left_child);
                nodes.push_back(right_child);
            }
        }
        true
    }
    pub fn height_2(root: &Option<Rc<RefCell<TreeNode>>>, height: usize) -> usize {
        match root {
            Some(root_of_tree) => {
                let left = &root_of_tree.borrow().left;
                let right = &root_of_tree.borrow().right;
                max(
                    Solution::height_2(left, height + 1),
                    Solution::height_2(right, height + 1),
                )
            }
            None => height,
        }
    }

    pub fn is_balanced(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        Solution::height_and_balance(&root, 0).is_balanced
    }
    pub fn height_and_balance(
        root: &Option<Rc<RefCell<TreeNode>>>,
        height: usize,
    ) -> HeightAndBalance {
        match root {
            Some(root_of_tree) => {
                let left = Solution::height_and_balance(&root_of_tree.borrow().left, height + 1);
                if !left.is_balanced {
                    return HeightAndBalance {
                        is_balanced: left.is_balanced,
                        height: left.height,
                    };
                }
                let right = Solution::height_and_balance(&root_of_tree.borrow().right, height + 1);
                if !right.is_balanced {
                    return HeightAndBalance {
                        is_balanced: right.is_balanced,
                        height: right.height,
                    };
                }
                let difference = (left.height as i32 - right.height as i32).abs();
                let is_balanced = left.is_balanced && right.is_balanced && difference <= 1;
                let tree_height = max(left.height, right.height);
                HeightAndBalance {
                    is_balanced,
                    height: tree_height,
                }
            }
            None => HeightAndBalance {
                height,
                is_balanced: true,
            },
        }
    }
}

struct HeightAndBalance {
    height: usize,
    is_balanced: bool,
}

impl Solution {
    pub fn climb_stairs(n: i32) -> i32 {
        let my_vec = vec![1; n as usize];
        let steps = Solution::find_steps(my_vec, n as usize);
        steps as i32
    }
    pub fn find_steps(nums: Vec<i32>, steps: usize) -> usize {
        println!("nums is {:?}", nums);
        if nums.len() == 1 {
            return steps;
        }
        let mut steps = steps;
        let mut pointer1 = 0;
        let mut pointer2 = 1;
        println!("steps is {}", steps);
        while pointer2 < nums.len() {
            if nums[pointer1] == 1 && nums[pointer2] == 1 {
                let mut my_vec = vec![1; nums.len() - 1];
                my_vec[pointer1] = 2;
                println!("my vec is {:?}", my_vec);
                let calculated_steps = Solution::find_steps(my_vec, steps + 1);
                steps += calculated_steps;
            }
            pointer1 += 1;
            pointer2 += 1;
        }
        steps
    }
}

use std::{any, i32, mem, usize};

use central_tendencies::heap::{GeneralHeap, Heap};
impl Solution {
    pub fn longest_palindrome(s: String) -> i32 {
        let mut letter_count = HashMap::new();
        for letter in s.chars() {
            let count = letter_count.entry(letter).or_insert(0);
            *count += 1;
        }
        let mut pivots = 0;
        let mut total_length = 0;
        for (_, value) in letter_count {
            if value % 2 == 0 {
                total_length += value;
            } else {
                if pivots == 0 {
                    pivots += 1;
                }
                total_length += value - 1;
            }
        }
        total_length + pivots
    }
}

impl Solution {
    pub fn reverse_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        Solution::traverse(None, head)
    }
    pub fn traverse(
        previous: Option<Box<ListNode>>,
        current: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        match current {
            Some(mut current_node) => {
                let next = current_node.next.take();
                current_node.next = previous;
                Solution::traverse(Some(current_node), next)
            }
            None => previous,
        }
    }
    pub fn reverse_list_iterative(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut current_node = head;
        let mut previous = None;
        while let Some(mut node) = current_node {
            let next = node.next.take();
            node.next = previous.take();
            previous = Some(node);
            current_node = next;
        }
        previous
    }
}

impl Solution {
    pub fn add_binary(a: String, b: String) -> String {
        let mut a_rev = a.chars().rev();
        let mut b_rev = b.chars().rev();
        let a_len = a.len();
        let b_len = b.len();

        let mut carry = 0;
        let mut result = String::new();

        for _ in 0..max(a_len, b_len) {
            let x = match a_rev.next() {
                Some(val) => val.to_digit(10).unwrap(),
                None => 0,
            };
            let y = match b_rev.next() {
                Some(val) => val.to_digit(10).unwrap(),
                None => 0,
            };

            let sum = x + y + carry;
            if sum == 3 {
                carry = 1;
                result.push('1');
            } else if sum == 2 {
                carry = 1;
                result.push('0');
            } else if sum == 1 {
                carry = 0;
                result.push('1');
            } else {
                carry = 0;
                result.push('0');
            }
        }
        if carry == 1 {
            result.push('1');
        }
        let result: String = result.chars().rev().collect();
        result
    }
}

impl Solution {
    pub fn diameter_of_binary_tree(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        let mut highest = 0;
        Solution::diameter(&root, &mut highest);
        highest as i32
    }
    pub fn diameter(root: &Option<Rc<RefCell<TreeNode>>>, highest: &mut usize) -> usize {
        match root {
            Some(root_of_tree) => {
                let left = &root_of_tree.borrow().left;
                let right = &root_of_tree.borrow().right;
                let left_height = Solution::diameter(left, highest);
                let right_height = Solution::diameter(right, highest);
                *highest = max(*highest, left_height + right_height);
                1 + max(left_height, right_height)
            }
            None => 0,
        }
    }
}

impl Solution {
    pub fn middle_node(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        // let head=head;
        let mut current = &head;
        let mut count = 0;
        while let Some(node) = current {
            count += 1;
            current = &node.next;
        }
        let mut current = head;
        if count % 2 == 0 {
            count = (count / 2) + 1;
        } else {
            count /= 2;
        }
        for _ in 0..count {
            if let Some(node) = current {
                current = node.next;
            }
        }

        current
    }
}

impl Solution {
    pub fn contains_duplicate(nums: Vec<i32>) -> bool {
        let mut visited = HashSet::with_capacity(nums.len());
        for num in nums {
            let is_inserted = visited.insert(num);
            if !is_inserted {
                return true;
            }
        }
        false
    }
}

impl Solution {
    pub fn max_sub_array(nums: Vec<i32>) -> i32 {
        let mut largest_sum = nums[0];
        let mut sum = 0;
        for num in nums {
            sum += num;
            if num > sum {
                sum = num;
            }
            largest_sum = max(largest_sum, sum);
        }
        largest_sum
    }
}

impl Solution {
    pub fn insert(intervals: Vec<Vec<i32>>, new_interval: Vec<i32>) -> Vec<Vec<i32>> {
        // let mut new_intervals=Vec::with_capacity(intervals.len()*2);
        let mut new_interval = new_interval;
        let mut final_intervals = Vec::new();
        for i in 0..intervals.len() {
            if new_interval[1] < intervals[i][0] {
                final_intervals.push(new_interval);
                final_intervals.extend_from_slice(&intervals[i..]);
                return final_intervals;
            }
            if new_interval[0] > intervals[i][1] {
                final_intervals.push(intervals[i].clone());
            } else {
                new_interval[0] = min(new_interval[0], intervals[i][0]);
                new_interval[1] = max(new_interval[1], intervals[i][1]);
            }
        }
        final_intervals.push(new_interval);
        final_intervals
    }
}

impl Solution {
    pub fn update_matrix(mat: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut mat = mat;
        let rows = mat.len();
        let columns = mat[0].len();
        let mut traversal_list = VecDeque::new();

        for i in 0..rows {
            for j in 0..columns {
                if mat[i][j] == 0 {
                    traversal_list.push_back((i, j));
                } else {
                    mat[i][j] = i32::MAX;
                }
            }
        }
        let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        while let Some((current_row_index, current_column_index)) = traversal_list.pop_front() {
            for (row_change, column_change) in directions {
                let neighbour_row_index = current_row_index as i32 + row_change;
                if neighbour_row_index < 0 || neighbour_row_index >= rows as i32 {
                    continue;
                }
                let neighbour_column_index = current_column_index as i32 + column_change;
                if neighbour_column_index < 0 || neighbour_column_index >= columns as i32 {
                    continue;
                }
                let neighbour_row_index = neighbour_row_index as usize;
                let neighbour_column_index = neighbour_column_index as usize;
                if mat[current_row_index][current_column_index] + 1
                    < mat[neighbour_row_index][neighbour_column_index]
                {
                    mat[neighbour_row_index][neighbour_column_index] =
                        mat[current_row_index][current_column_index] + 1;
                    traversal_list.push_back((neighbour_row_index, neighbour_column_index));
                }
            }
        }

        mat
    }
    fn find_neighbours(
        rows: usize,
        columns: usize,
        row_index: usize,
        column_index: usize,
    ) -> Vec<(usize, usize)> {
        let mut neighbours = Vec::new();
        let column_index_plus_1 = column_index + 1;
        if column_index_plus_1 < columns {
            neighbours.push((row_index, column_index_plus_1));
        }
        if let Some(column_index_minus_1) = column_index.checked_sub(1) {
            neighbours.push((row_index, column_index_minus_1));
        }
        let row_index_plus_1 = row_index + 1;
        if row_index_plus_1 < rows {
            neighbours.push((row_index_plus_1, column_index));
        }
        if let Some(row_index_minus_1) = row_index.checked_sub(1) {
            neighbours.push((row_index_minus_1, column_index));
        }
        neighbours
    }
}

impl Solution {
    pub fn k_closest(points: Vec<Vec<i32>>, k: i32) -> Vec<Vec<i32>> {
        let mut points_data_list = Vec::new();
        for point in points {
            let distance = point[0].pow(2) + point[1].pow(2);
            points_data_list.push(PointData {
                coordinates: point,
                distance,
            });
        }
        let mut points_heap = GeneralHeap::new(points_data_list);
        let mut final_results = Vec::new();
        for _ in 0..k {
            match points_heap.extract_min() {
                Some(val) => final_results.push(val.coordinates),
                None => break,
            }
        }
        final_results
    }
    pub fn k_closest_std_sort(points: Vec<Vec<i32>>, k: i32) -> Vec<Vec<i32>> {
        let mut points_data_list = Vec::new();
        for (index, point) in points.iter().enumerate() {
            let distance = point[0].pow(2) + point[1].pow(2);
            points_data_list.push(PointData_2 { index, distance });
        }
        points_data_list.sort_by_key(|a| a.distance);
        points_data_list
            .into_iter()
            .take(k as usize)
            .map(|point_data| points[point_data.index].clone())
            .collect()
    }
}

struct PointData {
    coordinates: Vec<i32>,
    distance: i32,
}

impl PartialEq for PointData {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl PartialOrd for PointData {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}
struct PointData_2 {
    index: usize,
    distance: i32,
}

impl Solution {
    pub fn length_of_longest_substring(s: String) -> i32 {
        let mut visited_hashmap = HashMap::new();
        let mut count = 0;
        let mut largest = 0;
        let mut last_interruption = 0;
        for (index, letter) in s.chars().enumerate() {
            match visited_hashmap.get(&letter) {
                Some(existing) => {
                    let existing_index = *existing;
                    largest = max(largest, count);
                    if last_interruption > existing_index {
                        count = index - last_interruption;
                    } else {
                        count = index - existing_index;
                        last_interruption = existing_index;
                    }
                    visited_hashmap.insert(letter, index);
                }
                None => {
                    visited_hashmap.insert(letter, index);
                    count += 1;
                }
            }
        }
        largest = max(largest, count);
        largest as i32
    }
}
impl Solution {
    pub fn three_sum_1(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut visited_hash_map = HashMap::new();
        let mut final_result = Vec::new();
        for (index, num) in nums.iter().enumerate() {
            let hash_set_ref = visited_hash_map.entry(*num).or_insert(HashSet::new());
            hash_set_ref.insert(index);
        }
        for i in 0..nums.len() - 1 {
            let sum = nums[i] + nums[i + 1];
            let differentiator = 0 - sum;
            if let Some(visited_indexes) = visited_hash_map.get(&differentiator) {
                if visited_indexes.contains(&i) || visited_indexes.contains(&(i + 1)) {
                    continue;
                }
                final_result.push(vec![nums[i], nums[i + 1], differentiator])
            }
        }
        final_result
    }
}

impl Solution {
    pub fn three_sum_2(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut nums = nums;
        let mut final_results = Vec::new();
        let nums_len = nums.len();
        nums.sort();
        for (index, num) in nums.iter().enumerate() {
            let target_for_two_sum = 0 - num;
            for inner_index in index + 1..nums_len {
                let inner_num = nums[inner_index];
                let complement = target_for_two_sum - inner_num;
                if complement < inner_num {
                    match Solution::binary_search(&nums[..inner_index], complement) {
                        Some(complementary_index) => {
                            if index != complementary_index {
                                final_results.push(vec![
                                    nums[index],
                                    nums[inner_index],
                                    nums[complementary_index],
                                ]);
                            }
                        }
                        None => continue,
                    }
                } else {
                    match Solution::binary_search(&nums[inner_index + 1..], complement) {
                        Some(complementary_index) => {
                            if index != complementary_index + inner_index + 1 {
                                final_results.push(vec![
                                    nums[index],
                                    nums[inner_index],
                                    nums[complementary_index + inner_index + 1],
                                ]);
                            }
                        }
                        None => continue,
                    }
                }
            }
        }
        final_results
    }
    pub fn binary_search(list: &[i32], value: i32) -> Option<usize> {
        if list.is_empty() {
            return None;
        }
        let mut low = 0;
        let mut high = list.len() - 1;

        while low <= high {
            let mid = low + ((high - low) / 2);
            if list[mid] == value {
                return Some(mid);
            }
            if value < list[mid] {
                if mid == 0 {
                    break;
                }
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        None
    }
}

impl Solution {
    pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut nums = nums;
        nums.sort();
        let mut result = Vec::new();
        for i in 0..nums.len() {
            if nums[i] > 0 {
                break;
            }
            if i > 0 && nums[i] == nums[i - 1] {
                continue;
            }
            Solution::two_sum_for_3_sum(&mut result, nums[i], &nums[i + 1..]);
        }

        result
    }

    pub fn two_sum_for_3_sum(result: &mut Vec<Vec<i32>>, precursor: i32, numbers: &[i32]) {
        if numbers.len() <= 1 {
            return;
        }
        let target = 0 - precursor;
        let mut left = 0;
        let mut right = numbers.len() - 1;
        while left < right {
            let sum = numbers[left] + numbers[right];
            if sum == target {
                result.push(vec![precursor, numbers[left], numbers[right]]);
                if left >= numbers.len() - 1 {
                    break;
                }
                left += 1;
                right -= 1;
                while left < right && numbers[left] == numbers[left - 1] {
                    left += 1;
                }
                while left < right && numbers[right] == numbers[right + 1] {
                    right -= 1;
                }
                continue;
            }
            if sum > target {
                right -= 1;
            } else {
                left += 1;
            }
        }
    }

    pub fn two_sum_ii(numbers: Vec<i32>, target: i32) -> Vec<i32> {
        let mut left = 0;
        let mut right = numbers.len() - 1;
        while left <= right {
            let sum = numbers[left] + numbers[right];
            if sum == target {
                return vec![(left + 1) as i32, (right + 1) as i32];
            }
            if sum > target {
                right -= 1;
            } else {
                left += 1;
            }
        }
        vec![]
    }
}

impl Solution {
    pub fn level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        let mut traversal_queue = VecDeque::new();
        traversal_queue.push_back(root);
        let mut result = Vec::new();

        while !traversal_queue.is_empty() {
            let mut level = Vec::new();
            for _ in 0..traversal_queue.len() {
                let node = traversal_queue.pop_front().unwrap();
                if let Some(current_node) = node {
                    level.push(current_node.borrow().val);
                    traversal_queue.push_back(current_node.borrow_mut().left.take());
                    traversal_queue.push_back(current_node.borrow_mut().right.take());
                }
            }
            if !level.is_empty() {
                result.push(level);
            }
        }
        result
    }
}

impl Solution {
    pub fn eval_rpn(tokens: Vec<String>) -> i32 {
        let operations = HashMap::from([
            ("+".to_string(), true),
            ("*".to_string(), true),
            ("-".to_string(), true),
            ("/".to_string(), true),
        ]);
        let mut operands: Vec<i32> = Vec::new();
        for maths_op in tokens.iter() {
            if operations.contains_key(maths_op) {
                let second_operand: i32 = operands.pop().unwrap();
                let first_operand: i32 = operands.pop().unwrap();
                if maths_op == "+" {
                    let result = first_operand + second_operand;
                    operands.push(result);
                } else if maths_op == "*" {
                    let result = first_operand * second_operand;
                    operands.push(result);
                } else if maths_op == "-" {
                    let result = first_operand - second_operand;
                    operands.push(result);
                } else {
                    let result = first_operand / second_operand;

                    operands.push(result);
                }
                continue;
            }
            operands.push(maths_op.parse::<i32>().unwrap());
        }
        operands[0]
    }
}

impl Solution {
    pub fn can_finish(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
        let mut adjacency_list: HashMap<i32, Vec<i32>> = HashMap::new();
        for course_and_prerequisite_combination in prerequisites {
            let course = course_and_prerequisite_combination[0];
            let prerequisite = course_and_prerequisite_combination[1];
            let neighbours = adjacency_list.entry(course).or_default();
            neighbours.push(prerequisite);
        }

        for course in 0..num_courses {
            let mut visited = HashSet::<i32>::new();
            let mut traversal_list = Vec::new();
            let mut ready = vec![false];
            traversal_list.push(course);
            while let Some(current_vertex) = traversal_list.pop() {
                let current_vertex_ready = ready.pop().unwrap();
                if current_vertex_ready {
                    adjacency_list.remove(&current_vertex);
                } else if let Some(neighbours) = adjacency_list.get(&current_vertex) {
                    if visited.contains(&current_vertex) {
                        return false;
                    }
                    visited.insert(current_vertex);
                    traversal_list.push(current_vertex);
                    ready.push(true);
                    for neighbour in neighbours {
                        if adjacency_list.contains_key(neighbour) {
                            traversal_list.push(*neighbour);
                            ready.push(false);
                        }
                    }
                }
            }
        }

        true
    }
}

fn nth_fibonnaci(n: usize) -> usize {
    let mut results = HashMap::<usize, usize>::new();
    results.insert(0, 0);
    results.insert(1, 1);
    let mut traversal_list = Vec::new();
    let mut ready = Vec::new();
    ready.push(false);
    traversal_list.push(n);
    while let Some(current) = traversal_list.pop() {
        let is_ready = ready.pop().unwrap();
        if results.contains_key(&current) {
            continue;
        }
        let number_minus_1 = current - 1;
        let number_minus_2 = current - 2;
        if is_ready {
            let sum = results.get(&number_minus_1).unwrap() + results.get(&number_minus_2).unwrap();
            results.insert(current, sum);
        } else {
            traversal_list.push(current);
            ready.push(true);

            if number_minus_2 > 1 {
                traversal_list.push(number_minus_2);
                ready.push(false);
            }
            if number_minus_1 > 1 {
                traversal_list.push(number_minus_1);
                ready.push(false);
            }
        }
    }
    *results.get(&n).unwrap()
}

fn fibonnaci_recursive(n: usize, calculated_results: &mut HashMap<usize, usize>) -> usize {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    match calculated_results.get(&n) {
        Some(hit) => *hit,
        None => {
            let result = fibonnaci_recursive(n - 1, calculated_results)
                + fibonnaci_recursive(n - 2, calculated_results);
            calculated_results.insert(n, result);
            result
        }
    }
}

impl Solution {
    pub fn coin_change(coins: Vec<i32>, amount: i32) -> i32 {
        let mut least_coins = vec![amount + 1; (amount + 1) as usize];
        least_coins[0] = 0;
        for i in 1..=amount as usize {
            let mut minimum = least_coins[i];
            for coin in coins.iter() {
                let sub_problem = match i.checked_sub((*coin) as usize) {
                    Some(val) => val,
                    None => continue,
                };
                minimum = min(minimum, least_coins[sub_problem]);
            }
            least_coins[i] = 1 + minimum;
        }

        if least_coins[(amount + 1) as usize] > amount {
            -1
        } else {
            least_coins[(amount + 1) as usize]
        }
    }
}

impl Solution {
    pub fn product_except_self(nums: Vec<i32>) -> Vec<i32> {
        let mut prefix_array = vec![1; nums.len()];
        let mut prefix = 1;
        let mut postfix = 1;

        for (index, num) in nums.iter().enumerate() {
            prefix_array[index] = prefix * num;
            prefix *= num;
        }
        let mut index = (nums.len() - 1) as i32;
        while index >= 0 {
            let i = index as usize;
            if i > 0 {
                prefix_array[i] = postfix * prefix_array[i - 1];
                postfix *= nums[i];
                index -= 1;
            } else {
                prefix_array[i] = postfix;
                index -= 1;
            }
        }
        prefix_array
    }
}

struct MinStackNode {
    val: i32,
    min_index_at_push: usize,
}
struct MinStack {
    list: Vec<i32>,
    minimum_list: Vec<i32>,
}

impl MinStack {
    fn new() -> Self {
        Self {
            list: Vec::new(),
            minimum_list: Vec::new(),
        }
    }

    fn push(&mut self, val: i32) {
        match self.minimum_list.last() {
            Some(minimum) => {
                self.list.push(val);
                self.minimum_list.push(*minimum.min(&val));
            }
            None => {
                self.list.push(val);
                self.minimum_list.push(val);
            }
        }
    }

    fn pop(&mut self) {
        self.list.pop();
        self.minimum_list.pop();
    }

    fn top(&self) -> i32 {
        *self.list.last().unwrap()
    }

    fn get_min(&self) -> i32 {
        *self.minimum_list.last().unwrap()
    }
}

struct MinStack2 {
    list: Vec<MinStackNode>,
    minimum: Option<i32>,
    min_index: Option<usize>,
}
impl MinStack2 {
    fn new() -> Self {
        Self {
            list: Vec::new(),
            min_index: None,
            minimum: None,
        }
    }

    fn push(&mut self, val: i32) {
        match self.minimum {
            Some(minimum) => {
                if val < minimum {
                    self.list.push(MinStackNode {
                        val,
                        min_index_at_push: self.list.len(),
                    });
                    self.min_index = Some(self.list.len() - 1);
                    self.minimum = Some(val);
                } else {
                    self.list.push(MinStackNode {
                        val,
                        min_index_at_push: self.min_index.unwrap(),
                    });
                }
            }
            None => {
                self.list.push(MinStackNode {
                    val,
                    min_index_at_push: 0,
                });
                self.min_index = Some(0);
                self.minimum = Some(val);
            }
        }
    }

    fn pop(&mut self) {
        if let Some(min_index) = self.min_index {
            if min_index == self.list.len() - 1 {
                self.list.pop();
                if !self.list.is_empty() {
                    let new_min_index = self.list[min_index - 1].min_index_at_push;
                    let new_minium = self.list[new_min_index].val;
                    self.min_index = Some(new_min_index);
                    self.minimum = Some(new_minium);
                } else {
                    self.min_index = None;
                    self.minimum = None;
                }
            } else {
                self.list.pop();
            }
        }
    }

    fn top(&self) -> i32 {
        self.list.last().unwrap().val
    }

    fn get_min(&self) -> i32 {
        self.minimum.unwrap()
    }
}

impl Solution {
    pub fn is_valid_bst(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        Solution::is_valid_binary_search_tree_helper(root, None, None)
    }
    pub fn is_valid_binary_search_tree_helper(
        root: Option<Rc<RefCell<TreeNode>>>,
        min: Option<i32>,
        max: Option<i32>,
    ) -> bool {
        match root {
            Some(valid_root) => {
                let left = valid_root.borrow_mut().left.take();
                let right = valid_root.borrow_mut().right.take();
                let node_value = valid_root.borrow().val;
                if let Some(min_val) = min {
                    if node_value <= min_val {
                        return false;
                    }
                }
                if let Some(max_val) = max {
                    if node_value >= max_val {
                        return false;
                    }
                }
                let is_left_child_valid =
                    Solution::is_valid_binary_search_tree_helper(left, min, Some(node_value));
                let is_right_child_valid =
                    Solution::is_valid_binary_search_tree_helper(right, Some(node_value), max);

                is_left_child_valid && is_right_child_valid
            }
            None => true,
        }
    }
}

impl Solution {
    pub fn num_islands(grid: Vec<Vec<char>>) -> i32 {
        let mut grid = grid;
        let rows = grid.len();
        let columns = grid[0].len();
        let mut no_of_islands = 0;

        for i in 0..rows {
            for j in 0..columns {
                if grid[i][j] == '1' {
                    no_of_islands += 1;
                    Solution::mark_island_parts(&mut grid, i, j, rows, columns);
                }
            }
        }
        no_of_islands
    }
    fn is_valid_pos_to_explore(
        rows: usize,
        columns: usize,
        row_index: i32,
        column_index: i32,
        grid: &Vec<Vec<char>>,
    ) -> bool {
        if row_index >= rows as i32 {
            return false;
        }
        if row_index < 0 {
            return false;
        }
        if column_index >= columns as i32 {
            return false;
        }
        if column_index < 0 {
            return false;
        }
        grid[row_index as usize][column_index as usize] == '1'
    }

    fn mark_island_parts(
        grid: &mut Vec<Vec<char>>,
        row_index: usize,
        column_index: usize,
        rows: usize,
        columns: usize,
    ) {
        grid[row_index][column_index] = '2';

        let possible_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        for (row_direction, column_direction) in possible_directions {
            let neigbor_row_index = row_index as i32 + row_direction;
            let neigbor_column_index = column_index as i32 + column_direction;
            if Solution::is_valid_pos_to_explore(
                rows,
                columns,
                neigbor_row_index,
                neigbor_column_index,
                grid,
            ) {
                Solution::mark_island_parts(
                    grid,
                    neigbor_row_index as usize,
                    neigbor_column_index as usize,
                    rows,
                    columns,
                );
            }
        }
    }
}

impl Solution {
    pub fn oranges_rotting(grid: Vec<Vec<i32>>) -> i32 {
        let mut grid = grid;
        let rows = grid.len();
        let columns = grid[0].len();
        let mut count_of_fresh_oranges = 0;

        let mut starting_rotten_oranges = Vec::new();
        for i in 0..rows {
            for j in 0..columns {
                if grid[i][j] == 2 {
                    starting_rotten_oranges.push((i, j));
                }
                if grid[i][j] == 1 {
                    count_of_fresh_oranges += 1;
                }
            }
        }
        if starting_rotten_oranges.is_empty() {
            if count_of_fresh_oranges > 0 {
                return -1;
            } else {
                return 0;
            }
        }
        let minutes_to_rot = Solution::make_nearby_oranges_rotten(
            &mut grid,
            &mut count_of_fresh_oranges,
            starting_rotten_oranges,
            rows,
            columns,
        );
        if count_of_fresh_oranges == 0 {
            minutes_to_rot
        } else {
            -1
        }
    }
    fn is_valid_ripe_orange_to_explore(
        rows: usize,
        columns: usize,
        row_index: i32,
        column_index: i32,
        grid: &[Vec<i32>],
    ) -> bool {
        if row_index >= rows as i32 {
            return false;
        }
        if row_index < 0 {
            return false;
        }
        if column_index >= columns as i32 {
            return false;
        }
        if column_index < 0 {
            return false;
        }
        grid[row_index as usize][column_index as usize] == 1
    }

    fn make_nearby_oranges_rotten(
        grid: &mut Vec<Vec<i32>>,
        count_of_fresh_oranges: &mut i32,
        starting_rotten_oranges: Vec<(usize, usize)>,
        rows: usize,
        columns: usize,
    ) -> i32 {
        let mut minutes = 0;
        let mut traversal_queue = VecDeque::from(starting_rotten_oranges);
        let possible_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        while !traversal_queue.is_empty() && *count_of_fresh_oranges > 0 {
            for _ in 0..traversal_queue.len() {
                let (row_index, column_index) = traversal_queue.pop_front().unwrap();
                for (row_direction, column_direction) in possible_directions {
                    let neigbor_row_index = row_index as i32 + row_direction;
                    let neigbor_column_index = column_index as i32 + column_direction;
                    if Solution::is_valid_ripe_orange_to_explore(
                        rows,
                        columns,
                        neigbor_row_index,
                        neigbor_column_index,
                        grid,
                    ) {
                        grid[neigbor_row_index as usize][neigbor_column_index as usize] = 3;
                        traversal_queue
                            .push_back((neigbor_row_index as usize, neigbor_column_index as usize));
                        *count_of_fresh_oranges -= 1;
                    }
                }
            }
            minutes += 1;
        }

        minutes
    }
}

impl Solution {
    pub fn search(nums: Vec<i32>, target: i32) -> i32 {
        let nums_len = nums.len();
        let mut left_pointer = 0;
        let mut right_pointer = nums_len - 1;

        while left_pointer < right_pointer {
            let min_index = left_pointer + (right_pointer - left_pointer) / 2;

            if nums[min_index] > nums[right_pointer] {
                left_pointer = min_index + 1;
            } else {
                right_pointer = min_index;
            }
        }
        let pivot = left_pointer;
        if pivot == 0 {
            left_pointer = 0;
            right_pointer = nums_len - 1;
        } else if target >= nums[0] && target <= nums[pivot - 1] {
            left_pointer = 0;
            right_pointer = pivot - 1;
        } else {
            left_pointer = pivot;
            right_pointer = nums_len - 1;
        }
        while left_pointer <= right_pointer {
            let mid = left_pointer + (right_pointer - left_pointer) / 2;
            if nums[mid] == target {
                return mid as i32;
            }
            if target < nums[mid] {
                if mid == 0 {
                    break;
                }
                right_pointer = mid - 1;
            } else {
                left_pointer = mid + 1;
            }
        }
        -1
    }
}

impl Solution {
    pub fn combination_sum(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut paths = Vec::new();
        let mut path = Vec::new();
        for (index, candidate) in candidates.iter().enumerate() {
            let difference = target - candidate;
            if difference < 0 {
                continue;
            }
            Solution::possible_paths(
                &mut path,
                &mut paths,
                (difference, *candidate),
                &candidates[index..],
            );
        }
        paths
    }
    fn possible_paths(
        path: &mut Vec<i32>,
        paths: &mut Vec<Vec<i32>>,
        node: (i32, i32),
        candidates: &[i32],
    ) {
        path.push(node.1);
        if node.0 == 0 {
            paths.push(path.clone());
        } else {
            for (index, candidate) in candidates.iter().enumerate() {
                let difference = node.0 - candidate;
                if difference < 0 {
                    continue;
                }
                Solution::possible_paths(
                    path,
                    paths,
                    (difference, *candidate),
                    &candidates[index..],
                );
            }
        }
        path.pop();
    }
}

impl Solution {
    pub fn permute(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut paths = Vec::new();
        let mut path = Vec::new();
        let mut visited = HashSet::new();
        for num in nums.iter() {
            Solution::possible_paths_permutations(&mut path, &mut paths, &mut visited, *num, &nums);
        }
        paths
    }
    fn possible_paths_permutations(
        path: &mut Vec<i32>,
        paths: &mut Vec<Vec<i32>>,
        visited: &mut HashSet<i32>,
        node: i32,
        nums: &[i32],
    ) {
        path.push(node);
        visited.insert(node);
        if path.len() == nums.len() {
            paths.push(path.clone());
        } else {
            for num in nums.iter() {
                if visited.contains(num) {
                    continue;
                }
                Solution::possible_paths_permutations(path, paths, visited, *num, nums);
            }
        }
        path.pop();
        visited.remove(&node);
    }
}

impl Solution {
    pub fn merge(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut intervals = intervals;
        intervals.sort_by_key(|a| a[0]);
        let mut final_intervals = Vec::new();
        final_intervals.push(intervals[0].clone());
        for current in intervals.into_iter().skip(1) {
            let last = final_intervals.last_mut().unwrap();
            if current[0] < last[1] {
                last[1] = max(current[1], last[1]);
            } else {
                final_intervals.push(current);
            }
        }
        final_intervals
    }
}

impl Solution {
    pub fn lowest_common_ancestor(
        root: Option<Rc<RefCell<TreeNode>>>,
        p: Option<Rc<RefCell<TreeNode>>>,
        q: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        match root {
            Some(root_node) => match p.zip(q) {
                Some((p_node, q_node)) => {
                    if root_node.borrow().val == p_node.borrow().val
                        || root_node.borrow().val == q_node.borrow().val
                    {
                        return Some(root_node);
                    }
                    let left = Solution::lowest_common_ancestor(
                        root_node.borrow().left.clone(),
                        Some(Rc::clone(&p_node)),
                        Some(Rc::clone(&q_node)),
                    );
                    let right = Solution::lowest_common_ancestor(
                        root_node.borrow().right.clone(),
                        Some(p_node),
                        Some(q_node),
                    );
                    if left.is_some() && right.is_some() {
                        return Some(root_node);
                    }
                    left.or(right)
                }
                None => None,
            },
            None => None,
        }
    }
    pub fn lowest_common_ancestor_through_traversal(
        root: Option<Rc<RefCell<TreeNode>>>,
        p: Option<Rc<RefCell<TreeNode>>>,
        q: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        let mut path1 = Vec::new();
        let mut path2 = Vec::new();
        match root {
            Some(root_node) => {
                match p {
                    Some(p_node) => {
                        Solution::traverse_tree(
                            &mut path1,
                            Rc::clone(&root_node),
                            Rc::clone(&p_node),
                        );
                    }
                    None => {
                        return None;
                    }
                }
                match q {
                    Some(q_node) => {
                        Solution::traverse_tree(
                            &mut path2,
                            Rc::clone(&root_node),
                            Rc::clone(&q_node),
                        );
                    }
                    None => {
                        return None;
                    }
                }
                let path1_len = path1.len();
                let path2_len = path2.len();
                let mut current = Rc::clone(&path1[0]);
                let mut path1 = path1.into_iter();
                let mut path2 = path2.into_iter();
                for _ in 0..max(path1_len, path2_len) {
                    match path1.next() {
                        Some(current_node) => match path2.next() {
                            Some(current_node_path2) => {
                                if current_node.borrow().val != current_node_path2.borrow().val {
                                    return Some(current);
                                }
                                current = current_node_path2;
                            }
                            None => return Some(current),
                        },
                        None => return Some(current),
                    }
                }
                Some(current)
            }
            None => None,
        }
    }

    fn traverse_tree(
        path: &mut Vec<Rc<RefCell<TreeNode>>>,
        node: Rc<RefCell<TreeNode>>,
        target: Rc<RefCell<TreeNode>>,
    ) -> bool {
        path.push(Rc::clone(&node));
        if node.borrow().val == target.borrow().val {
            return true;
        }
        let left = node.borrow().left.clone();
        let right = node.borrow().right.clone();
        let mut is_in_right = false;
        let mut is_in_left = false;
        if left.is_some() {
            is_in_left = Solution::traverse_tree(path, left.unwrap(), Rc::clone(&target));
        }
        if right.is_some() {
            is_in_right = Solution::traverse_tree(path, right.unwrap(), Rc::clone(&target));
        }
        if is_in_left || is_in_right {
            return true;
        }
        path.pop();
        false
    }
}

struct TimeMap2 {
    map: HashMap<String, BTreeMap<i32, String>>,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl TimeMap2 {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn set(&mut self, key: String, value: String, timestamp: i32) {
        self.map
            .entry(key)
            .and_modify(|existing| {
                existing
                    .entry(timestamp)
                    .and_modify(|existing_time_map| {
                        *existing_time_map = value.clone();
                    })
                    .or_insert(value.clone());
            })
            .or_insert(BTreeMap::from([(timestamp, value)]));
    }

    fn get(&self, key: String, timestamp: i32) -> String {
        match self.map.get(&key) {
            Some(existing) => match existing.get(&timestamp) {
                Some(existing_time_map) => existing_time_map.to_string(),
                None => {
                    for (key, value) in existing.iter().rev() {
                        if key < &timestamp {
                            return value.to_string();
                        }
                    }
                    String::new()
                }
            },
            None => String::new(),
        }
    }
}

struct TimeMap {
    map: HashMap<String, Vec<(i32, String)>>,
}
impl TimeMap {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
    fn set(&mut self, key: String, value: String, timestamp: i32) {
        self.map
            .entry(key)
            .and_modify(|existing| {
                existing.push((timestamp, value.clone()));
            })
            .or_insert(vec![(timestamp, value)]);
    }
    fn get(&self, key: String, timestamp: i32) -> String {
        match self.map.get(&key) {
            Some(existing) => {
                let mut left_pointer = 0;
                let mut right_pointer = existing.len() - 1;
                let mut result = String::new();
                while left_pointer <= right_pointer {
                    let mid = left_pointer + (right_pointer - left_pointer) / 2;
                    if existing[mid].0 <= timestamp {
                        if existing[mid].0 == timestamp {
                            return existing[mid].1.to_string();
                        }
                        result = existing[mid].1.to_string();
                        left_pointer = mid + 1;
                    } else {
                        if mid == 0 {
                            break;
                        }
                        right_pointer = mid - 1;
                    }
                }
                result
            }
            None => String::new(),
        }
    }
}

#[derive(Debug)]
struct Account {
    name: String,
    emails: HashSet<String>,
    active: bool,
}

impl Solution {
    pub fn accounts_merge_2(accounts: Vec<Vec<String>>) -> Vec<Vec<String>> {
        let mut email_map = HashMap::new();
        let mut final_result: Vec<Account> = Vec::new();
        let mut merged_accounts: HashMap<usize, usize> = HashMap::new();
        let mut merged = HashSet::new();

        for (index, account) in accounts.iter().enumerate() {
            for address in account.iter().skip(1) {
                if let Some(previous_index) = email_map.insert(address, index) {
                    if previous_index == index {
                        continue;
                    } else {
                        let mut previous_should_be_updated = false;
                        let mut current_should_be_updated = false;

                        let mut previous_to_change = previous_index;
                        let mut current_to_change = index;
                        let mut others_need_change = false;

                        let index_in_final = match merged_accounts.get(&previous_index) {
                            Some(merged_index_in_final) => match merged_accounts.get(&index) {
                                Some(current_index_in_final) => {
                                    if merged_index_in_final == current_index_in_final {
                                        continue;
                                    }
                                    if final_result[*merged_index_in_final].active {
                                        let second =
                                            final_result.get_mut(*current_index_in_final).unwrap();
                                        second.active = false;
                                        let emails = second.emails.clone();
                                        final_result[*merged_index_in_final].emails.extend(emails);
                                        current_should_be_updated = true;
                                        current_to_change = *current_index_in_final;
                                        others_need_change = true;
                                        *merged_index_in_final
                                    } else {
                                        let second =
                                            final_result.get_mut(*merged_index_in_final).unwrap();
                                        second.active = false;
                                        let emails = second.emails.clone();
                                        final_result[*current_index_in_final].emails.extend(emails);
                                        previous_should_be_updated = true;
                                        previous_to_change = *merged_index_in_final;
                                        others_need_change = true;
                                        *current_index_in_final
                                    }
                                }
                                None => {
                                    let second: Vec<String> = accounts[index]
                                        .iter()
                                        .skip(1)
                                        .filter(|a| *a != address)
                                        .cloned()
                                        .collect();
                                    final_result[*merged_index_in_final]
                                        .emails
                                        .extend(second.into_iter());
                                    merged.insert(index);
                                    current_should_be_updated = true;
                                    *merged_index_in_final
                                }
                            },
                            None => match merged_accounts.get(&index) {
                                Some(current_index_in_final) => {
                                    let second: Vec<String> = accounts[previous_index]
                                        .iter()
                                        .skip(1)
                                        .filter(|a| *a != address)
                                        .cloned()
                                        .collect();
                                    final_result[*current_index_in_final]
                                        .emails
                                        .extend(second.into_iter());
                                    merged.insert(previous_index);
                                    previous_should_be_updated = true;
                                    *current_index_in_final
                                }
                                None => {
                                    let mut first: Vec<String> =
                                        accounts[previous_index].iter().skip(1).cloned().collect();
                                    let second: Vec<String> = accounts[index]
                                        .iter()
                                        .skip(1)
                                        .filter(|a| *a != address)
                                        .cloned()
                                        .collect();
                                    first.extend_from_slice(&second);
                                    final_result.push(Account {
                                        name: accounts[previous_index][0].clone(),
                                        emails: first.into_iter().collect(),
                                        active: true,
                                    });
                                    merged.insert(previous_index);
                                    merged.insert(index);
                                    previous_should_be_updated = true;
                                    current_should_be_updated = true;
                                    final_result.len() - 1
                                }
                            },
                        };
                        if current_should_be_updated {
                            if others_need_change {
                                for (key, value) in merged_accounts.iter_mut() {
                                    if *value == current_to_change {
                                        *value = index_in_final;
                                    }
                                }
                            }
                            merged_accounts.insert(index, index_in_final);
                        }
                        if previous_should_be_updated {
                            if others_need_change {
                                for (key, value) in merged_accounts.iter_mut() {
                                    if *value == previous_to_change {
                                        *value = index_in_final;
                                    }
                                }
                            }
                            merged_accounts.insert(previous_index, index_in_final);
                        }
                        println!(
                            "index is {} previous index is {} address is {} merged_accounts {:?}",
                            index, previous_index, address, merged_accounts
                        );
                        println!("final result is {:?}", final_result);
                    }
                }
            }
        }

        for i in 0..accounts.len() {
            if merged.contains(&i) {
                continue;
            }
            final_result.push(Account {
                name: accounts[i][0].clone(),
                emails: accounts[i][1..].iter().cloned().collect(),
                active: true,
            });
        }
        println!("final result is {:?}", final_result);
        let mut final_result: Vec<Vec<String>> = final_result
            .into_iter()
            .filter(|a| a.active)
            .map(|a| {
                let mut vec_form = vec![a.name];
                vec_form.extend(a.emails);
                vec_form
            })
            .collect();
        for item in final_result.iter_mut() {
            let sub_slice = &mut item[1..];
            sub_slice.sort();
        }
        final_result.sort_by(|a, b| a[0].cmp(&b[0]));
        final_result
    }
}

impl Solution {
    pub fn accounts_merge(accounts: Vec<Vec<String>>) -> Vec<Vec<String>> {
        let mut disjoint_set = UnionFind::new(accounts.len());
        let mut email_map = HashMap::new();
        for (index, account) in accounts.iter().enumerate() {
            for address in account.iter().skip(1) {
                if let Some(existing_index) = email_map.insert(address, index) {
                    if existing_index == index {
                        continue;
                    } else {
                        disjoint_set.union(existing_index, index);
                    }
                }
            }
        }
        let mut accounts_map = HashMap::new();
        for (key, value) in email_map {
            let root = disjoint_set.find(value);
            accounts_map
                .entry(root)
                .and_modify(|a: &mut Vec<String>| a.push(key.to_string()))
                .or_insert(vec![key.to_string()]);
        }
        let mut final_result = Vec::new();
        for (key, mut value) in accounts_map {
            let mut account = vec![accounts[key][0].clone()];
            value.sort();
            account.append(&mut value);
            final_result.push(account);
        }
        final_result
    }
}

impl Solution {
    pub fn sort_colors_bucket_sort(nums: &mut Vec<i32>) {
        let mut buckets: Vec<i32> = vec![0; 3];

        for num in nums.iter() {
            buckets[*num as usize] += 1;
        }
        let mut i = 0;
        for (index, bucket) in buckets.iter_mut().enumerate() {
            while *bucket > 0 {
                nums[i] = index as i32;
                i += 1;
                *bucket -= 1;
            }
        }
    }
}

impl Solution {
    pub fn sort_colors(nums: &mut Vec<i32>) {
        let mut left_pointer = 0;
        let mut right_pointer = nums.len() - 1;
        let mut i = 0;

        while i <= right_pointer {
            if nums[i] == 0 {
                nums.swap(left_pointer, i);
                left_pointer += 1;
                i += 1;
            } else if nums[i] == 2 {
                nums.swap(right_pointer, i);
                right_pointer -= 1;
            } else {
                i += 1;
            }
        }
    }
}

impl Solution {
    pub fn word_break_wrong(s: String, word_dict: Vec<String>) -> bool {
        let mut intervals = Vec::new();
        for word in word_dict {
            let word_len = word.len();
            let mut start = 1;
            while let Some(position) = &s[start..].find(&word) {
                let starting_interval = position + start - 1;
                let closing_interval = starting_interval + word_len - 1;
                intervals.push((starting_interval, closing_interval));
                start = closing_interval + 1;
            }
        }
        intervals.sort_by(|a, b| a.0.cmp(&b.0));
        for (index, interval) in intervals.iter().enumerate() {
            let index_plus_1 = index + 1;
            if index_plus_1 >= intervals.len() {
                break;
            }
            if interval.1 > intervals[index + 1].0 {
                return false;
            }
        }
        true
    }
}

impl Solution {
    pub fn word_break(s: String, word_dict: Vec<String>) -> bool {
        let s_len = s.len();
        let mut word_break = vec![false; s_len + 1];
        word_break[s_len] = true;

        let mut i = (s_len - 1) as i32;
        while i >= 0 {
            let index = i as usize;
            for word in word_dict.iter() {
                if index + word.len() < s.len() && &s[index..index + word.len()] == word {
                    word_break[index] = word_break[index + word.len()];
                }
                if word_break[index] {
                    break;
                }
            }
            i -= 1;
        }
        word_break[0]
    }
}

impl Solution {
    pub fn can_partition_recursion_with_memoization(nums: Vec<i32>) -> bool {
        let sum: i32 = nums.iter().sum();
        if sum % 2 == 1 {
            return false;
        }
        let target = sum / 2;
        let mut calculated: HashMap<(usize, i32), bool> = HashMap::new();

        Solution::find_subset_sum(target, &nums, 0, 0, &mut calculated)
    }
    pub fn find_subset_sum(
        target: i32,
        nums: &Vec<i32>,
        index: usize,
        sum: i32,
        cache: &mut HashMap<(usize, i32), bool>,
    ) -> bool {
        if sum > target {
            return false;
        }
        if index >= nums.len() {
            return false;
        }
        if sum == target {
            return true;
        }
        if let Some(value) = cache.get(&(index, sum)) {
            return *value;
        }
        let take = Solution::find_subset_sum(target, nums, index + 1, sum + nums[index], cache);
        if take {
            cache.insert((index, sum), take);
            return take;
        }
        let not_take = Solution::find_subset_sum(target, nums, index + 1, sum, cache);
        let has_sum = take || not_take;
        cache.insert((index, sum), has_sum);
        take || not_take
    }
}

impl Solution {
    pub fn can_partition_dp(nums: Vec<i32>) -> bool {
        let sum: i32 = nums.iter().sum();
        if sum % 2 == 1 {
            return false;
        }
        let target = sum / 2;
        let mut sums = HashSet::<i32>::with_capacity(target as usize);
        sums.insert(0);

        for num in nums.into_iter().rev() {
            let mut new_sums = Vec::new();
            for sum in sums.iter() {
                let new_sum = num + sum;
                if new_sum == target {
                    return true;
                }
                new_sums.push(new_sum);
            }
            sums.extend(new_sums.iter());
        }

        false
    }
}

impl Solution {
    pub fn can_partition(nums: Vec<i32>) -> bool {
        let sum: i32 = nums.iter().sum();
        if sum % 2 == 1 {
            return false;
        }
        let target = (sum / 2) as usize;
        let mut sum_dp = vec![false; target + 1];
        if nums.iter().any(|&n| n as usize == target) {
            return true;
        }
        if nums[0] <= target as i32 {
            sum_dp[nums[0] as usize] = true;
        }
        for num in nums.into_iter().skip(1) {
            let num = num as usize;
            if num > target {
                continue;
            }
            sum_dp[target] = sum_dp[target - num] || sum_dp[target];
            if sum_dp[target] {
                return true;
            }
            for index in (num..target).rev() {
                sum_dp[index] = sum_dp[index - num] || sum_dp[index];
            }
        }
        sum_dp[target]
    }
}

impl Solution {
    pub fn my_atoi(s: String) -> i32 {
        let mut split_text = s.split_whitespace();
        let mut numbers = String::new();
        let mut sign_found = false;

        let first_item = match split_text.next() {
            Some(val) => val,
            None => return 0,
        };
        let chars = first_item.chars();
        for character in chars {
            if character.is_numeric() {
                numbers.push(character);
            } else if character == '-' || character == '+' {
                if !sign_found && numbers.is_empty() {
                    numbers.push(character);
                    sign_found = true;
                } else {
                    let final_number: i32 = match numbers.parse() {
                        Ok(val) => val,
                        Err(err) => match err.kind() {
                            std::num::IntErrorKind::PosOverflow => {
                                return i32::MAX;
                            }
                            std::num::IntErrorKind::NegOverflow => {
                                return i32::MIN;
                            }

                            _ => return 0,
                        },
                    };
                    return final_number;
                }
            } else {
                let final_number: i32 = match numbers.parse() {
                    Ok(val) => val,
                    Err(err) => match err.kind() {
                        std::num::IntErrorKind::PosOverflow => {
                            return i32::MAX;
                        }
                        std::num::IntErrorKind::NegOverflow => {
                            return i32::MIN;
                        }

                        _ => return 0,
                    },
                };
                return final_number;
            }
        }

        let final_number: i32 = match numbers.parse() {
            Ok(val) => val,
            Err(err) => match err.kind() {
                std::num::IntErrorKind::PosOverflow => {
                    return i32::MAX;
                }
                std::num::IntErrorKind::NegOverflow => {
                    return i32::MIN;
                }

                _ => return 0,
            },
        };
        final_number
    }
}

impl Solution {
    pub fn spiral_order(matrix: Vec<Vec<i32>>) -> Vec<i32> {
        let m = matrix.len();
        let n = matrix[0].len();
        let mut previous_horizontal_movement: i32 = 1;
        let mut horizontal_movement: i32 = 1;
        let mut final_result = Vec::new();
        let mut vertical_movement: i32 = 0;
        let mut i: i32 = 0;
        let mut j: i32 = 0;
        let mut previous_vertical_movement: i32 = -1;
        let mut visited_no = 0;
        let mut visited: HashSet<(i32, i32)> = HashSet::new();
        while visited_no < (m * n) {
            visited.insert((i, j));
            if horizontal_movement != 0 {
                final_result.push(matrix[i as usize][j as usize]);
                j += horizontal_movement;
                visited_no += 1;
                if j >= n as i32 || j < 0 || visited.contains(&(i, j)) {
                    println!("n is {n} and j is {j}");
                    j -= horizontal_movement;
                    previous_horizontal_movement = horizontal_movement;
                    horizontal_movement = 0;
                    vertical_movement = -previous_vertical_movement;
                    i += vertical_movement;
                }
            } else if vertical_movement != 0 {
                println!("j is {j} i is {i}");
                final_result.push(matrix[i as usize][j as usize]);
                i += vertical_movement;
                visited_no += 1;
                if i >= m as i32 || i < 0 || visited.contains(&(i, j)) {
                    i -= vertical_movement;
                    previous_vertical_movement = vertical_movement;
                    vertical_movement = 0;
                    horizontal_movement = -previous_horizontal_movement;
                    j += horizontal_movement;
                }
            }
            println!("final result {:?}", final_result);
        }
        final_result
    }
}

impl Solution {
    pub fn subsets(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut paths = Vec::new();
        let mut path = Vec::new();
        Solution::find_paths(0, &nums, &mut paths, &mut path, true);
        Solution::find_paths(0, &nums, &mut paths, &mut path, false);
        paths
    }
    pub fn find_paths(
        index: usize,
        nums: &[i32],
        paths: &mut Vec<Vec<i32>>,
        path: &mut Vec<i32>,
        include_current_number: bool,
    ) {
        if index >= nums.len() {
            return;
        }
        if include_current_number {
            path.push(nums[index]);
        }
        if index + 1 >= nums.len() {
            paths.push(path.clone());
        }
        Solution::find_paths(index + 1, nums, paths, path, true);
        Solution::find_paths(index + 1, nums, paths, path, false);
        if include_current_number {
            path.pop();
        }
    }
}

impl Solution {
    pub fn right_side_view_2(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        let mut visible = Vec::new();
        let mut outermost_right_side_depth =
            Solution::find_depth_of_side_tree(root.clone(), 0, &mut visible);
        Solution::traverse_binary_tree_2(root, 1, &mut visible, &mut outermost_right_side_depth);
        visible
    }
    pub fn find_depth_of_side_tree(
        tree_node: Option<Rc<RefCell<TreeNode>>>,
        depth: usize,
        visible: &mut Vec<i32>,
    ) -> usize {
        match tree_node {
            Some(node) => {
                let right = node.borrow().right.clone();
                visible.push(node.borrow().val);
                if right.is_some() {
                    Solution::find_depth_of_side_tree(right, depth + 1, visible)
                } else {
                    let left = node.borrow_mut().left.clone();
                    Solution::find_depth_of_side_tree(left, depth + 1, visible)
                }
            }
            None => depth,
        }
    }
    pub fn traverse_binary_tree_2(
        tree_node: Option<Rc<RefCell<TreeNode>>>,
        depth: usize,
        visible: &mut Vec<i32>,
        right_side_depth: &mut usize,
    ) {
        if let Some(node) = tree_node {
            let node = Rc::clone(&node);
            let left = node.borrow().left.clone();
            let right = node.borrow().right.clone();
            if depth > *right_side_depth {
                visible.push(node.borrow().val);
                *right_side_depth = depth;
            }
            Solution::traverse_binary_tree_2(right, depth + 1, visible, right_side_depth);
            Solution::traverse_binary_tree_2(left, depth + 1, visible, right_side_depth);
        }
    }
}

impl Solution {
    pub fn right_side_view_dfs(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        let mut visible = Vec::new();
        let mut outermost_right_side_depth = 0;
        Solution::traverse_binary_tree_dfs(root, 1, &mut visible, &mut outermost_right_side_depth);
        visible
    }

    pub fn traverse_binary_tree_dfs(
        tree_node: Option<Rc<RefCell<TreeNode>>>,
        depth: usize,
        visible: &mut Vec<i32>,
        right_side_depth: &mut usize,
    ) {
        if let Some(node) = tree_node {
            let node = Rc::clone(&node);
            let left = node.borrow().left.clone();
            let right = node.borrow().right.clone();
            if depth > *right_side_depth {
                visible.push(node.borrow().val);
                *right_side_depth = depth;
            }
            Solution::traverse_binary_tree_dfs(right, depth + 1, visible, right_side_depth);
            Solution::traverse_binary_tree_dfs(left, depth + 1, visible, right_side_depth);
        }
    }
}

impl Solution {
    pub fn right_side_view_bfs(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        let mut visible = Vec::new();
        let mut traversal = VecDeque::new();
        match root {
            Some(val) => {
                traversal.push_back(val);
            }
            None => {
                return visible;
            }
        }

        while !traversal.is_empty() {
            let initial_len = traversal.len();
            let node = traversal.pop_front().unwrap();
            println!("val is {} len is {}", node.borrow().val, initial_len);
            let right = node.borrow_mut().right.take();
            let left = node.borrow_mut().left.take();
            if let Some(val) = right {
                traversal.push_back(val);
            }
            if let Some(val) = left {
                traversal.push_back(val);
            }
            visible.push(node.borrow().val);
            for _ in 0..initial_len - 1 {
                let node = traversal.pop_front().unwrap();
                let right = node.borrow_mut().right.take();
                let left = node.borrow_mut().left.take();
                if let Some(val) = right {
                    traversal.push_back(val);
                }
                if let Some(val) = left {
                    traversal.push_back(val);
                }
            }
        }
        visible
    }
}


impl Solution {
    pub fn longest_palindromic_substring(s: String) -> String {
        let chars:Vec<char>=s.chars().collect();
        let chars_len=chars.len();
        let mut result=(chars_len,chars_len);
        for i in 0..chars_len{
            let mut left=i;
            let mut right=i;
            while right<chars_len&&chars[left]==chars[right]{
                
                if (right-left)>(result.1-result.0){
                    result=(left,right);
                    if (right-left+1)==chars_len{
                        return chars[result.0..=result.1].into_iter().collect();
                        
                    }
                }
                if left==0{
                    break;

                }
                left-=1;
                right+=1;

            }
            let mut left=i;
            let mut right=i+1;
            while right<chars_len&&chars[left]==chars[right]{
                if (right-left)>(result.1-result.0){
                    result=(left,right);
                }
                if left==0{
                    break;

                }
                left-=1;
                right+=1;

            }

        }
        if result.0==chars_len{
            return String::from(chars[0]);
        }
        let longest:String=chars[result.0..=result.1].into_iter().collect();
        longest
        
        
    }
    pub fn find_sub_strings(
        index: usize,
        chars: &[char],
        paths: &mut Vec<String>,
        path: &mut Vec<char>,
        include_current_number: bool,
    ) {
        if index >= chars.len() {
            return;
        }
        if include_current_number {
            path.push(chars[index]);
        }
        if index + 1 >= chars.len() {
            paths.push(path.iter().collect());
        }
        Solution::find_sub_strings(index + 1, chars, paths, path, true);
        Solution::find_sub_strings(index + 1, chars, paths, path, false);
        if include_current_number {
            path.pop();
        }
    }

    pub fn valid_palindrome(word:&str)->bool{
        if word.is_empty(){
            return false;
        }
        let chars:Vec<char>=word.chars().map(|a|a.to_ascii_uppercase()).collect();
        let mut right=chars.len()-1;
        let mut left=0;

        while left<right{
            if chars[left]!=chars[right]{
                return false;
            }
            left+=1;
            right-=1;
        }
        true

    }
        
    
}


impl Solution {
    pub fn unique_paths(m: i32, n: i32) -> i32 {
        let mut possible_paths=vec![1;n as usize];
        let mut i=m-2;
        while i>=0{
            let mut j=n-2;
            while j>=0{
                let mut no_of_paths=0;
                if j+1<n{
                    no_of_paths+=possible_paths[(j+1) as usize];
                }
                if i+1<m{
                    no_of_paths+=possible_paths[j as usize];
                }
                possible_paths[j as usize]=no_of_paths;
                j-=1;


            }
            i-=1;
            


        }
        possible_paths[0]
        
    }
}
