use super::lexer::{Token, TokenKind};
use super::parsing_symbols::*;

pub fn is_a_substring_of_b(a: &[char], b: &[char]) -> bool {
    if b.len() >= a.len() && a.len() != 0 {
        for i in 0..a.len() {
            if a[i] != b[i] {
                return false;
            }
        }
        return true;
    }
    false
}

pub fn is_whitespace(raw: &[char]) -> Option<usize> {
    let whitespace: Vec<Vec<char>> = WHITESPACE
        .iter()
        .map(|keyword| keyword.chars().collect::<Vec<char>>())
        .collect();
    for s in whitespace {
        if is_a_substring_of_b(&s, raw) {
            return Some(s.len());
        }
    }
    None
}

pub fn is_number(raw: &[char]) -> Option<usize> {
    if raw.len() > 0 {
        let mut i = 0;
        while raw[i].is_digit(10) {
            i += 1;
        }
        if i > 0 {
            return Some(i);
        }
    }
    None
}

pub fn is_identifier(raw: &[char]) -> Option<usize> {
    if raw.len() > 0 {
        if !raw[0].is_digit(10) {
            let mut i = 0;
            while i < raw.len() && (raw[i].is_alphanumeric() || raw[i] == '_') {
                i += 1;
            }
            if i > 0 {
                return Some(i);
            }
        }
    }
    None
}

pub fn is_keyword(raw: &[char]) -> Option<usize> {
    let keywords: Vec<Vec<char>> = KEYWORDS
        .iter()
        .map(|keyword| keyword.chars().collect::<Vec<char>>())
        .collect();

    for keyword in keywords {
        if is_a_substring_of_b(&keyword, raw) {
            return Some(keyword.len());
        }
    }
    None
}

pub fn is_syntax(raw: &[char]) -> Option<usize> {
    if raw.len() > 0 {
        for s in SYNTAX {
            if raw[0] == s {
                return Some(1);
            }
        }
    }
    None
}

pub fn is_right_associative(raw: &[char]) -> bool {
    match String::from_iter(raw).as_str() {
        OP_POW => return true,
        _ => return false,
    }
}

pub fn operator_precedence(raw: &[char]) -> usize {
    match String::from_iter(raw).as_str() {
        OP_EQ | OP_NEQ => 0,
        OP_LT | OP_LTE | OP_GT | OP_GTE => 10,
        OP_SHL | OP_SHR => 20,
        OP_PLUS | OP_MINUS => 30,
        OP_MUL | OP_DIV | OP_MOD => 40,
        OP_POW => 50,
        _ => panic!("Invalid operator."),
    }
}

pub fn is_operator(raw: &[char]) -> Option<usize> {
    let operators: Vec<Vec<char>> = OPERATORS
        .iter()
        .map(|keyword| keyword.chars().collect::<Vec<char>>())
        .collect();

    for operator in operators {
        if is_a_substring_of_b(&operator, raw) {
            return Some(operator.len());
        }
    }
    None
}

pub fn expect_operator(tokens: &[Token<'_>], index: usize, value: &str) -> bool {
    if index >= tokens.len() {
        return false;
    }
    let p = value.chars().collect::<Vec<char>>();
    tokens[index].kind() == TokenKind::Operator
        && p.len() == tokens[index].value().len()
        && is_a_substring_of_b(tokens[index].value(), &p)
}

pub fn expect_keyword(tokens: &[Token<'_>], index: usize, value: &str) -> bool {
    if index >= tokens.len() {
        return false;
    }
    let p = value.chars().collect::<Vec<char>>();
    tokens[index].kind() == TokenKind::Keyword
        && p.len() == tokens[index].value().len()
        && is_a_substring_of_b(tokens[index].value(), &p)
}

pub fn expect_syntax(tokens: &[Token<'_>], index: usize, value: &str) -> bool {
    if index >= tokens.len() {
        return false;
    }
    let p = value.chars().collect::<Vec<char>>();
    tokens[index].kind() == TokenKind::Syntax
        && p.len() == tokens[index].value().len()
        && is_a_substring_of_b(tokens[index].value(), &p)
}

pub fn expect_identifier(tokens: &[Token<'_>], index: usize) -> bool {
    if index >= tokens.len() {
        return false;
    }
    tokens[index].kind() == TokenKind::Identifier
}
