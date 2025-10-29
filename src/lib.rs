pub mod heap;
pub mod sorting;
pub mod graph;
pub mod tree;
pub mod union_find;


#[cfg(test)]
mod trie_tests {
    use crate::tree::Trie;
    #[test]
    fn test_insert_and_search() {
        let mut trie = Trie::default();
        
        // Insert words
        trie.insert("hello".to_string());
        trie.insert("world".to_string());
        trie.insert("hell".to_string());
        
        // Search for existing words
        assert!(trie.search("hello"));
        assert!(trie.search("world"));
        assert!(trie.search("hell"));
        
        // Search for non-existing words
        assert!(!trie.search("hel"));
        assert!(!trie.search("worlds"));
        assert!(!trie.search("goodbye"));
    }

    #[test]
    fn test_starts_with() {
        let mut trie = Trie::default();
        
        trie.insert("apple".to_string());
        trie.insert("app".to_string());
        trie.insert("application".to_string());
        
        // Test valid prefixes
        assert!(trie.starts_with("app".to_string()));
        assert!(trie.starts_with("appl".to_string()));
        assert!(trie.starts_with("apple".to_string()));
        assert!(trie.starts_with("a".to_string()));
        
        // Test invalid prefixes
        assert!(!trie.starts_with("ban".to_string()));
        assert!(trie.starts_with("appli".to_string())); // Fixed: this should be true
        assert!(!trie.starts_with("b".to_string()));
    }

    #[test]
    fn test_prefix_words() {
        let mut trie = Trie::default();
        
        trie.insert("car".to_string());
        trie.insert("card".to_string());
        trie.insert("care".to_string());
        trie.insert("careful".to_string());
        
        // "car" is both a word and a prefix
        assert!(trie.search("car"));
        assert!(trie.starts_with("car".to_string()));
        
        // "card" exists
        assert!(trie.search("card"));
        
        // "care" exists
        assert!(trie.search("care"));
        
        // "careful" exists
        assert!(trie.search("careful"));
        
        // "ca" is a prefix but not a word
        assert!(!trie.search("ca"));
        assert!(trie.starts_with("ca".to_string()));
    }

    #[test]
    fn test_find_all_words() {
        let mut trie = Trie::default();
        
        trie.insert("cat".to_string());
        trie.insert("car".to_string());
        trie.insert("dog".to_string());
        
        let mut words = trie.find_all_words();
        words.sort(); // Sort for consistent comparison
        
        let mut expected = vec!["cat".to_string(), "car".to_string(), "dog".to_string()];
        expected.sort();
        
        assert_eq!(words, expected);
    }

    #[test]
    fn test_delete_leaf_word() {
        let mut trie = Trie::default();
        
        trie.insert("cat".to_string());
        trie.insert("car".to_string());
        
        assert!(trie.search("cat"));
        assert!(trie.search("car"));
        
        // Delete "cat"
        trie.delete("cat".to_string());
        
        assert!(!trie.search("cat"));
        assert!(trie.search("car")); // "car" should still exist
        assert!(trie.starts_with("ca".to_string())); // "ca" prefix should still exist
    }

    #[test]
    fn test_delete_prefix_word() {
        let mut trie = Trie::default();
        
        trie.insert("car".to_string());
        trie.insert("card".to_string());
        
        assert!(trie.search("car"));
        assert!(trie.search("card"));
        
        // Delete "car" (which is a prefix of "card")
        trie.delete("car".to_string());
        
        assert!(!trie.search("car"));
        assert!(trie.search("card")); // "card" should still exist
        assert!(trie.starts_with("car".to_string())); // "car" prefix should still exist
    }

    #[test]
    fn test_delete_word_with_shared_prefix() {
        let mut trie = Trie::default();
        
        trie.insert("cat".to_string());
        trie.insert("car".to_string());
        trie.insert("dog".to_string());
        
        // Delete "cat"
        trie.delete("cat".to_string());
        
        assert!(!trie.search("cat"));
        assert!(trie.search("car")); // "car" should still exist
        assert!(trie.search("dog")); // "dog" should still exist
    }

    #[test]
    fn test_delete_only_word() {
        let mut trie = Trie::default();
        
        trie.insert("hello".to_string());
        
        assert!(trie.search("hello"));
        
        trie.delete("hello".to_string());
        
        assert!(!trie.search("hello"));
        assert!(!trie.starts_with("hello".to_string()));
        assert!(trie.starts_with("h".to_string()));
    }

    #[test]
    fn test_delete_sub_string_of_word() {
        let mut trie = Trie::default();
        
        trie.insert("book".to_string());

        trie.insert("bookmark".to_string());
        
        assert!(trie.search("book"));

        assert!(trie.search("bookmark"));
        
        trie.delete("book".to_string());
        
        assert!(!trie.search("book"));
        assert!(trie.search("bookmark"));
    }

    #[test]
    fn test_delete_super_string_of_word() {
        let mut trie = Trie::default();
        
        trie.insert("book".to_string());

        trie.insert("bookmark".to_string());
        
        assert!(trie.search("book"));

        assert!(trie.search("bookmark"));
        
        trie.delete("bookmark".to_string());
        
        assert!(trie.search("book"));
        assert!(!trie.search("bookmark"));
    }
    #[test]
    fn test_delete_one_path_of_many_paths() {
        let mut trie = Trie::default();
        
        trie.insert("mini".to_string());
        trie.insert("minimal".to_string());
        trie.insert("minimum".to_string());
        trie.insert("minimally".to_string());
        let all_words=trie.find_all_words();
        assert_eq!(all_words,vec!["mini".to_string(),"minimal".to_string(),"minimally".to_string(),"minimum".to_string()]);    
        assert!(trie.search("minimal"));
        assert!(trie.search("minimum"));
        assert!(trie.search("mini"));
        assert!(trie.search("minimally"));
        trie.delete("minimally".to_string());
        let all_words=trie.find_all_words();
        assert_eq!(all_words,vec!["mini".to_string(),"minimal".to_string(),"minimum".to_string()]); 
        
    }

    #[test]
    fn autocomplete() {
        let mut trie = Trie::default();
        
        trie.insert("mini".to_string());
        trie.insert("minimal".to_string());
        trie.insert("minimum".to_string());
        trie.insert("minimally".to_string());
        let mut possible_matches=trie.find_possible_matches("mini".to_string());
        possible_matches.sort();
        assert_eq!(possible_matches,vec!["mini".to_string(),"minimal".to_string(),"minimally".to_string(),"minimum".to_string()]);
        let mut possible_matches=trie.find_possible_matches("minim".to_string());
        possible_matches.sort();
        assert_eq!(possible_matches,vec!["minimal".to_string(),"minimally".to_string(),"minimum".to_string()]);
        let mut possible_matches=trie.find_possible_matches("minimal".to_string());
        possible_matches.sort();
        assert_eq!(possible_matches,vec!["minimal".to_string(),"minimally".to_string()]);
        let mut possible_matches=trie.find_possible_matches("minimum".to_string());
        possible_matches.sort();
        assert_eq!(possible_matches,vec!["minimum".to_string()]);   
    }

    #[test]
    fn test_delete_non_existent_word() {
        let mut trie = Trie::default();
        
        trie.insert("cat".to_string());
        
        // Try to delete a word that doesn't exist
        trie.delete("dog".to_string());
        
        // "cat" should still exist
        assert!(trie.search("cat"));
    }

    #[test]
    fn test_empty_trie() {
        let trie = Trie::default();
        
        assert!(!trie.search("anything"));
        assert!(!trie.starts_with("any".to_string()));
        
        let words = trie.find_all_words();
        assert!(words.is_empty());
    }

    #[test]
    fn test_single_character_words() {
        let mut trie = Trie::default();
        
        trie.insert("a".to_string());
        trie.insert("i".to_string());
        
        assert!(trie.search("a"));
        assert!(trie.search("i"));
        assert!(!trie.search("b"));
        
        trie.delete("a".to_string());
        
        assert!(!trie.search("a"));
        assert!(trie.search("i"));
    }

    #[test]
    fn test_long_words() {
        let mut trie = Trie::default();
        
        let long_word = "pneumonoultramicroscopicsilicovolcanoconiosis".to_string();
        trie.insert(long_word.clone());
        
        assert!(trie.search(&long_word));
        assert!(trie.starts_with("pneumono".to_string()));
        
        trie.delete(long_word.clone());
        assert!(!trie.search(&long_word));
    }

    #[test]
    fn test_special_characters() {
        let mut trie = Trie::default();
        
        trie.insert("hello-world".to_string());
        trie.insert("hello_world".to_string());
        
        assert!(trie.search("hello-world"));
        assert!(trie.search("hello_world"));
        assert!(!trie.search("hello world"));
    }

    #[test]
    fn test_case_sensitivity() {
        let mut trie = Trie::default();
        
        trie.insert("Hello".to_string());
        trie.insert("hello".to_string());
        
        assert!(trie.search("Hello"));
        assert!(trie.search("hello"));
        assert!(!trie.search("HELLO"));
    }

    #[test]
    fn test_multiple_deletions() {
        let mut trie = Trie::default();
        
        let words = vec!["apple", "app", "application", "apply", "banana"];
        for word in &words {
            trie.insert(word.to_string());
        }
        
        // Delete in order
        trie.delete("apple".to_string());
        assert!(!trie.search("apple"));
        assert!(trie.search("app"));
        
        trie.delete("app".to_string());
        assert!(!trie.search("app"));
        assert!(trie.search("application"));
        
        trie.delete("application".to_string());
        assert!(!trie.search("application"));
        assert!(trie.search("apply"));
        
        trie.delete("apply".to_string());
        assert!(!trie.search("apply"));
        assert!(trie.search("banana"));
        
        trie.delete("banana".to_string());
        assert!(!trie.search("banana"));
    }

    #[test]
    fn test_find_all_words_after_deletions() {
        let mut trie = Trie::default();
        
        trie.insert("cat".to_string());
        trie.insert("car".to_string());
        trie.insert("card".to_string());
        trie.insert("dog".to_string());
        
        trie.delete("car".to_string());
        
        let mut words = trie.find_all_words();
        words.sort();
        
        let mut expected = vec!["cat".to_string(), "card".to_string(), "dog".to_string()];
        expected.sort();
        
        assert_eq!(words, expected);
    }
}
