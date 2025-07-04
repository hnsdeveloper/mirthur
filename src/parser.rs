use super::helpers::*;
use super::lexer::{Token, TokenKind};
use super::parsing_symbols::*;

type AST<'a> = Vec<Statement<'a>>;

pub enum Statement<'a> {
    Expression(Expression<'a>),
    If(If<'a>),
    FunctionDeclaration(FunctionDeclaration<'a>),
    Return(Return<'a>),
    Let(Let<'a>),
}

pub enum Literal<'a> {
    Identifier(Token<'a>),
    Number(Token<'a>),
    // TODO: Add string
}

pub struct FunctionCall<'a> {
    pub name: Token<'a>,
    pub arguments: Vec<Expression<'a>>,
}

pub struct UnaryOperation<'a> {
    pub operator: Token<'a>,
    pub exp: Box<Expression<'a>>,
}

pub struct BinaryOperation<'a> {
    pub operator: Token<'a>,
    pub left: Box<Expression<'a>>,
    pub right: Box<Expression<'a>>,
}

pub enum Expression<'a> {
    FunctionCall(FunctionCall<'a>),
    UnaryOperation(UnaryOperation<'a>),
    BinaryOperation(BinaryOperation<'a>),
    Literal(Literal<'a>),
}

pub struct FunctionDeclaration<'a> {
    pub name: Token<'a>,
    pub parameters: Vec<Token<'a>>,
    pub body: Vec<Statement<'a>>,
}

pub struct If<'a> {
    pub test: Box<Expression<'a>>,
    pub body: Vec<Statement<'a>>,
}

pub struct Let<'a> {
    pub name: Token<'a>,
    pub exp: Expression<'a>,
}

pub struct Return<'a> {
    pub exp: Option<Expression<'a>>,
}

pub struct Parser {}

impl Parser {
    pub fn parse<'a>(raw: &[char], tokens: &'a [Token]) -> Result<AST<'a>, String> {
        let mut ast = vec![];
        let mut i = 0;
        while i < tokens.len() {
            match parse_statement(raw, tokens, i) {
                Ok((statement, new_i)) => {
                    i = new_i;
                    ast.push(statement);
                }
                Err((a, b)) => {
                    let mut s = tokens[i]
                        .location()
                        .debug(raw, "Invalid token while parsing: ");
                    s.push_str(&format!("Expected {}, found {}.", a, b));
                    return Err(s);
                }
            }
        }
        Ok(ast)
    }
}

fn parse_statement<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    idx: usize,
) -> Result<(Statement<'a>, usize), (String, String)> {
    todo!();
}

fn parse_let<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    index: usize,
) -> Option<(Statement<'a>, usize)> {
    if expect_syntax(tokens, index, &KW_LET) {
        let id_idx = index + 1;
        if expect_identifier(tokens, id_idx) {
            let eq_idx = id_idx + 1;
            if expect_syntax(tokens, eq_idx, &SYNTAX_EQ.to_string()) {
                let exp_begin_idx = eq_idx + 1;
                if let Some((exp, idx)) = parse_expression(raw, tokens, exp_begin_idx, 0) {
                    let stmt = Statement::Let(Let {
                        name: tokens[id_idx].clone(),
                        exp,
                    });
                    let semicolon_idx = idx + 1;
                    if expect_syntax(tokens, semicolon_idx, &SYNTAX_SC.to_string()) {
                        return Some((stmt, semicolon_idx + 1));
                    }
                }
            }
        }
    }
    None
}

fn parse_expression_statement<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    index: usize,
) -> Option<(Statement<'a>, usize)> {
    if let Some((exp, i)) = parse_expression(raw, tokens, index, 0) {
        if expect_syntax(tokens, i, &SYNTAX_SC.to_string()) {
            return Some((Statement::Expression(exp), i + 1));
        }
    }
    None
}

fn parse_primary<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    index: usize,
) -> Option<(Expression<'a>, usize)> {
    if index < tokens.len() {
        if tokens[index].kind() == TokenKind::Syntax
            && expect_syntax(tokens, index, &SYNTAX_OP.to_string())
        {
            let exp_begin_idx = index + 1;
            if let Some((exp, i)) = parse_expression(raw, tokens, exp_begin_idx, 0) {
                if expect_syntax(tokens, i, &SYNTAX_CP.to_string()) {
                    return Some((exp, i + 1));
                }
            }
        } //else if tokens[index].kind() == TokenKind::Number && 

    }
    None
}

fn parse_expression<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    index: usize,
    min_precedence: usize,
) -> Option<(Expression<'a>, usize)> {
    if index < tokens.len() {}
    None
}
