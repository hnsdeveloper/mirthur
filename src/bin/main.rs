use bevy_lua::prelude::*;

use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    let contents = fs::read_to_string("test.mt").expect("Could not read file");

    let raw: Vec<char> = contents.chars().collect();

    let tokens = match Lexer::lex(&raw) {
        Ok(tokens) => tokens,
        Err(msg) => panic!("{}", msg),
    };

    for token in tokens {
        match token.kind() {
            TokenKind::Identifier => println!("{} : Identifier", String::from_iter(token.value())),
            TokenKind::Syntax => println!("{} : Syntax", String::from_iter(token.value())),
            TokenKind::Keyword => println!("{} : Keyword", String::from_iter(token.value())),
            TokenKind::Number => println!("{} : Number", String::from_iter(token.value())),
            TokenKind::Operator => println!("{} : Operator", String::from_iter(token.value())),
        }
    }
}
