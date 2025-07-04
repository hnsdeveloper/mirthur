use super::lexer::{Token, TokenKind};
use super::parsing_symbols::*;

pub fn a_biased_strcmp(a: &[char], b: &[char]) -> bool {
    if b.len() >= a.len() && a.len() > 0 && b.len() > 0 {
        let mut i = 0;
        while i < a.len() {
            if a[i] == b[i] {
                i += 1;
            } else {
                break;
            }
        }
        if i == a.len() {
            return true;
        }
    }
    false
}

pub fn is_whitespace(raw: &[char]) -> Option<usize> {
    let whitespace: Vec<Vec<char>> = WHITESPACE
        .iter()
        .map(|keyword| keyword.chars().collect::<Vec<char>>())
        .collect();
    for s in whitespace {
        if a_biased_strcmp(&s, raw) {
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
        if a_biased_strcmp(&keyword, raw) {
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
    todo!()
}

pub fn operator_precedence(raw: &[char]) -> usize {
    todo!()
}

pub fn is_operator(raw: &[char]) -> Option<usize> {
    let operators: Vec<Vec<char>> = OPERATORS
        .iter()
        .map(|keyword| keyword.chars().collect::<Vec<char>>())
        .collect();

    for operator in operators {
        if a_biased_strcmp(&operator, raw) {
            return Some(operator.len());
        }
    }
    None
}

pub fn is_token_operator(tokens: &[Token<'_>], index: usize) -> bool {
    for operator in OPERATORS {
        if expect_operator(tokens, index, operator) {
            return true;
        }
    }
    false
}

pub fn expect_operator(tokens: &[Token<'_>], index: usize, value: &str) -> bool {
    if index >= tokens.len() {
        return false;
    }
    tokens[index].kind() == TokenKind::Operator
        && a_biased_strcmp(tokens[index].value(), &value.chars().collect::<Vec<char>>())
}

pub fn expect_keyword(tokens: &[Token<'_>], index: usize, value: &str) -> bool {
    if index >= tokens.len() {
        return false;
    }
    tokens[index].kind() == TokenKind::Keyword
        && a_biased_strcmp(tokens[index].value(), &value.chars().collect::<Vec<char>>())
}

pub fn expect_syntax(tokens: &[Token<'_>], index: usize, value: &str) -> bool {
    if index >= tokens.len() {
        return false;
    }
    tokens[index].kind() == TokenKind::Syntax
        && a_biased_strcmp(tokens[index].value(), &value.chars().collect::<Vec<char>>())
}

pub fn expect_identifier(tokens: &[Token<'_>], index: usize) -> bool {
    if index >= tokens.len() {
        return false;
    }
    tokens[index].kind() == TokenKind::Identifier
}
