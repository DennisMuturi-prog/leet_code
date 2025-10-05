pub struct Heap {
    list: Vec<i32>,
}
impl Heap {
    pub fn new(list: Vec<i32>) -> Self {
        let mut heap = Heap { list };
        heap.heapify();
        heap
    }
    fn sift_up(&mut self, index: usize) {
        if index == 0 {
            return;
        }
        let mut index = index;

        while index > 0 {
            let parent = (index - 1) / 2;
            if self.list[index] < self.list[parent] {
                self.list.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }
    fn sift_down(&mut self, index: usize) {
        let mut index = index;
        let list_len = self.list.len();
        while index < self.list.len() {
            let left_child = (2 * index) + 1;
            let right_child = (2 * index) + 2;
            if right_child >= list_len && left_child >= list_len {
                break;
            }
            if right_child >= list_len
                && left_child < list_len
                && self.list[index] > self.list[left_child]
            {
                self.list.swap(index, left_child);
                break;
            }
            if left_child >= list_len
                && right_child < list_len
                && self.list[index] > self.list[right_child]
            {
                self.list.swap(index, right_child);
                break;
            }
            if left_child < list_len
                && right_child < list_len
                && (self.list[index] > self.list[left_child]
                    || self.list[index] > self.list[right_child])
            {
                let child_to_swap = if self.list[left_child] < self.list[right_child] {
                    left_child
                } else {
                    right_child
                };
                self.list.swap(index, child_to_swap);
                index = child_to_swap;
            } else {
                break;
            }
        }
    }
    fn heapify(&mut self) {
        if self.list.is_empty() {
            return;
        }
        let mut i = self.list.len() - 1;
        loop {
            self.sift_down(i);
            i = match i.checked_sub(1) {
                Some(val) => val,
                None => return,
            };
        }
    }
    fn extract_min(&mut self) -> Option<i32> {
        if self.list.is_empty() {
            return None;
        }
        let min = self.list[0];
        let last = self.list.len() - 1;
        self.list.swap(0, last);
        self.list.pop();
        self.sift_down(0);
        Some(min)
    }
    pub fn update(&mut self, old_value: i32, new_value: i32)->Option<()> {
        let mut pos=0;
        let mut found=false;
        for (index,value) in self.list.iter().enumerate(){
            if *value==old_value{
                pos=index;
                found=true;
            }
        }
        if !found{
            return None;
        }
        self.update_with_index(pos, new_value);
        Some(())
    }
    pub fn update_with_index(&mut self, index: usize, new_value: i32) {
        let old_value=self.list[index];
        if new_value>old_value{
            self.list[index]=new_value;
            self.sift_down(index);

        }else if new_value<old_value{
            self.list[index]=new_value;
            self.sift_up(index);

        }else{
            return;
        }
        println!("heap is {:?}",self.list);

    }
    pub fn get_sorted_list(&mut self) -> Vec<i32> {
        let mut sorted_list = Vec::new();
        while let Some(item) = self.extract_min() {
            sorted_list.push(item);
        }
        sorted_list
    }
}

pub struct GeneralHeap<T, F> {
    list: Vec<T>,
    key_function: F,
}
impl<T, F, K> GeneralHeap<T, F>
where
    F: FnMut(&T) -> K,
    K: Ord,
{
    pub fn new(list: Vec<T>, key_function: F) -> Self {
        let mut heap = GeneralHeap { list, key_function };
        heap.heapify();
        heap
    }
    fn sift_up(&mut self, index: usize) {
        let key_function=&mut self.key_function;
        if index == 0 {
            return;
        }
        let mut index = index;

        while index > 0 {
            let parent = (index - 1) / 2;
            if key_function(&self.list[index]) < key_function(&self.list[parent]) {
                self.list.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }
    fn sift_down(&mut self, index: usize) {
        let key_function=&mut self.key_function;
        let mut index = index;
        let list_len = self.list.len();
        while index < self.list.len() {
            let left_child = (2 * index) + 1;
            let right_child = (2 * index) + 2;
            if right_child >= list_len && left_child >= list_len {
                break;
            }
            if right_child >= list_len
                && left_child < list_len
                && key_function(&self.list[index]) > key_function(&self.list[left_child])
            {
                self.list.swap(index, left_child);
                break;
            }
            if left_child >= list_len
                && right_child < list_len
                && key_function(&self.list[index]) > key_function(&self.list[right_child])
            {
                self.list.swap(index, right_child);
                break;
            }
            if left_child < list_len
                && right_child < list_len
                && (key_function(&self.list[index]) > key_function(&self.list[left_child])
                    || key_function(&self.list[index]) > key_function(&self.list[right_child]))
            {
                let child_to_swap = if key_function(&self.list[left_child]) < key_function(&self.list[right_child]) {
                    left_child
                } else {
                    right_child
                };
                self.list.swap(index, child_to_swap);
                index = child_to_swap;
            } else {
                break;
            }
        }
    }
    fn heapify(&mut self) {
        if self.list.is_empty() {
            return;
        }
        let mut i = self.list.len() - 1;
        loop {
            self.sift_down(i);
            i = match i.checked_sub(1) {
                Some(val) => val,
                None => return,
            };
        }
    }
    pub fn extract_min(&mut self) -> Option<T> {
        if self.list.is_empty() {
            return None;
        }
        let last = self.list.len() - 1;
        self.list.swap(0, last);
        let min=self.list.pop();
        self.sift_down(0);
        min
    }
    pub fn update(&mut self, old_value: T, new_value: T)->Option<()> {
        let mut pos=0;
        let mut found=false;
        for (index,value) in self.list.iter().enumerate(){
            let key_function=&mut self.key_function;
            if key_function(value)==key_function(&old_value){
                pos=index;
                found=true;
            }
        }
        if !found{
            return None;
        }
        self.update_with_index(pos, new_value);
        Some(())
    }
    pub fn update_with_index(&mut self, index: usize, new_value: T) {
        let key_function=&mut self.key_function;
        let old_value=&self.list[index];
        if key_function(&new_value)>key_function(old_value){
            self.list[index]=new_value;
            self.sift_down(index);

        }else if key_function(&new_value)<key_function(old_value){
            self.list[index]=new_value;
            self.sift_up(index);

        }else{
            return;
        }

    }
    pub fn get_sorted_list(&mut self) -> Vec<T> {
        let mut sorted_list = Vec::new();
        while let Some(item) = self.extract_min() {
            sorted_list.push(item);
        }
        sorted_list
    }
}
