use std::collections::HashMap;
use central_tendencies::graph::{Graph,Neighbour};


fn main() {
     let adjacency_list = HashMap::from([
        (
            0,
            vec![
                Neighbour::new(2, 2.4),
                Neighbour::new(3, 2.2),
            ],
        ),
        (
            1,
            vec![
                Neighbour::new(3, 2.2),
                Neighbour::new(4, 2.5),
            ]
        ),
        (
            2,
            vec![
                Neighbour::new(0, 2.4),
                Neighbour::new(5, 2.0),
                Neighbour::new(6, 2.8),
            ],
        ),
        (
            3,
            vec![
                Neighbour::new(0, 2.2),
                Neighbour::new(1, 2.2),
                Neighbour::new(7, 3.1),
                Neighbour::new(6, 2.6),
                Neighbour::new(5, 3.4),
            ],
        ),
        (
            4,
            vec![
                Neighbour::new(1, 2.5),
                Neighbour::new(7, 2.1),
                Neighbour::new(6, 2.9),
            ],
        ),
        (
            5,
            vec![
                Neighbour::new(2, 2.0),
                Neighbour::new(3, 3.4),
                Neighbour::new(8, 2.8),
                Neighbour::new(9, 4.0),
            ],
        ),
        (
            6,
            vec![
                Neighbour::new(3, 2.6),
                Neighbour::new(2, 2.8),
                Neighbour::new(4, 2.9),
                Neighbour::new(8, 2.8),
                Neighbour::new(9, 2.4),
            ],
        ),
        (
            7,
            vec![
                Neighbour::new(3, 3.1),
                Neighbour::new(4, 2.1),
                Neighbour::new(8, 4.4),
                Neighbour::new(9, 2.6),
            ],
        ),
        (
            8,
            vec![
                Neighbour::new(5, 2.8),
                Neighbour::new(6, 2.8),
                Neighbour::new(7, 4.4),
            ],
        ),
        (
            9,
            vec![
                Neighbour::new(6, 2.4),
                Neighbour::new(7, 2.6),
                Neighbour::new(5, 4.0),
            ],
        ),
    ]);


    let mut graph = Graph::new(adjacency_list);
    let result = graph.shortest_path(5, 4);
    println!("result is {:?}", result);

    let adjacency_matrix = vec![
        vec![0.0, 0.0, 2.4, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 2.2, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![2.4, 0.0, 0.0, 0.0, 0.0, 2.0, 2.8, 0.0, 0.0, 0.0],
        vec![2.2, 2.2, 0.0, 0.0, 0.0, 3.4, 2.6, 3.1, 0.0, 0.0],
        vec![0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 2.9, 2.1, 0.0, 0.0],
        vec![0.0, 0.0, 2.0, 3.4, 0.0, 0.0, 0.0, 0.0, 2.8, 4.0],
        vec![0.0, 0.0, 2.8, 2.6, 2.9, 0.0, 0.0, 0.0, 2.8, 2.4],
        vec![0.0, 0.0, 0.0, 3.1, 2.1, 0.0, 0.0, 0.0, 4.4, 2.6],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.8, 2.8, 4.4, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.4, 2.6, 0.0, 0.0],
    ];

    let mut graph = Graph::try_from(adjacency_matrix).unwrap();
    let result = graph.shortest_path(0, 8);
    println!("result is {:?}", result);
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
use core::num;
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
    pub fn lowest_common_ancestor(
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

use std::{i32, usize};

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
                new_interval[1] = min(new_interval[1], intervals[i][1]);
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

impl PartialEq for PointData{
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl PartialOrd for PointData{
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
