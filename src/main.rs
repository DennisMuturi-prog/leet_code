fn main() {}

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
use std::cell::RefCell;
use std::cmp::max;
use std::collections::VecDeque;
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

use std::collections::HashMap;
use std::usize;
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
        let mut a_rev=a.chars().rev();
        let mut b_rev=b.chars().rev();
        let a_len = a.len();
        let b_len = b.len();
        
        let mut carry=0;
        let mut result=String::new();


        for _ in 0..max(a_len, b_len) {
          let x=match a_rev.next(){
            Some(val) => val.to_digit(10).unwrap(),
            None => 0,
          };
          let y=match b_rev.next(){
            Some(val) => val.to_digit(10).unwrap(),
            None => 0,
          };
          
          let sum=x+y+carry;
          if sum==3{
            carry=1;
            result.push('1');

          }else if sum==2{
            carry=1;
            result.push('0');

          }
          else if sum==1{
            carry=0;
            result.push('1');
            
          }else{
            carry=0;
            result.push('0');

          }
          
        }
        if carry==1{
          result.push('1');
        }
        let result:String=result.chars().rev().collect();
        result
    }
}


impl Solution {
    pub fn diameter_of_binary_tree(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
      let mut highest=0;
      Solution::diameter(&root, &mut highest);
      highest as i32

        
    }
    pub fn diameter(root: &Option<Rc<RefCell<TreeNode>>>,highest:&mut usize) -> usize {
        match root {
            Some(root_of_tree) => {
                let left = &root_of_tree.borrow().left;
                let right = &root_of_tree.borrow().right;
                let left_height=Solution::diameter(left,highest);
                let right_height=Solution::diameter(right,highest);
                *highest=max(*highest,left_height+right_height);
                1+max(
                    
                    left_height,right_height
                )
            }
            None => 0,
        }
    }
}
