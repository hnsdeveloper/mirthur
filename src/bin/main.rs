use bevy_lua::prelude::*;
use std::error::Error;
use std::fs;

fn main() -> Result<(), Box<dyn Error>> {
    let contents = fs::read_to_string("test.mt").expect("Could not read file");

    let raw: Vec<char> = contents.chars().collect();

    let tokens = match Lexer::lex(&raw) {
        Ok(tokens) => tokens,
        Err(msg) => panic!("{}", msg),
    };

    let ast = Parser::parse(&raw, &tokens).unwrap();
    let mut program = Program::build(Compiler::build(ast).compile().unwrap(), 1024 * 1024);
    let _ = program.run();

    Ok(())
}
