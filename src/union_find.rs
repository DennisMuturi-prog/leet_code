pub struct UnionFind{
    list:Vec<usize>,
    rank:Vec<usize>
}

impl UnionFind{
    pub fn new(len:usize)->Self{
        Self{
            list:(0..len).collect(),
            rank:vec![0;len]
        }
    }
    pub fn find(&mut self,x:usize)->usize{
        if self.list[x]==x{
            x
        }else{
            self.list[x]=self.find(self.list[x]);
            self.list[x]
        }
    }
    pub fn union(&mut self,x:usize,y:usize){
        let root_of_x=self.find(x);
        let root_of_y=self.find(y);
        if self.rank[root_of_x]<self.rank[root_of_y]{
            self.list[root_of_x]=root_of_y;
            
        }else if self.rank[root_of_y]<self.rank[root_of_x]{
            self.list[root_of_y]=root_of_x;

        }else{
            self.list[root_of_y]=root_of_x;
            self.rank[root_of_x]+=1;

        }
    }
}