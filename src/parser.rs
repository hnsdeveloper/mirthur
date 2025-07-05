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
    pub is_pre: bool,
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
    pub tests: Vec<Expression<'a>>,
    pub bodies: Vec<Vec<Statement<'a>>>,
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

// Inner body of function declarations and if statements.
fn parse_inner_body<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    index: usize,
) -> Option<(Vec<Statement<'a>>, usize)> {
    let mut v: Vec<_> = Vec::new();
    let mut i = index;

    while i < tokens.len() {}

    Some((v, i))
}

fn parse_if<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    mut index: usize,
) -> Option<(Statement<'a>, usize)> {
    if index < tokens.len() {
        if expect_keyword(tokens, index, "if") {
            let mut tests: Vec<Expression<'a>> = Vec::new();
            let mut bodies: Vec<Vec<Statement<'a>>> = Vec::new();
            index += 1;
            let (exp, i) = parse_expression(raw, tokens, index, 0)?;
            index = i;
            if !expect_keyword(tokens, index, "then") {
                return None;
            }
            index += 1;
            let (stmt, i) = parse_inner_body(raw, tokens, index)?;
            index = i;
            bodies.push(stmt);
            while index < tokens.len() {
                if expect_keyword(tokens, index, "elif") {
                    index += 1;
                    let (elif_test, i) = parse_expression(raw, tokens, index, 0)?;
                    index = i;
                    if !expect_keyword(tokens, index, "then") {
                        return None;
                    }
                    index += 1;
                    let (elif_stmt, i) = parse_inner_body(raw, tokens, index)?;
                    index = i;
                    tests.push(elif_test);
                    bodies.push(elif_stmt);
                    continue;
                }
                if expect_keyword(tokens, index, "else") {
                    index += 1;
                    index = i;
                    // If there is an else, we will have one extra statement without test. In the virtual machine, for it
                    // to work properly, we add a test that always returns > 0.
                    let (else_stmt, i) = parse_inner_body(raw, tokens, index)?;
                    bodies.push(else_stmt);
                }
                if expect_keyword(tokens, index, "end") {
                    return Some((Statement::If(If { tests, bodies }), index + 1));
                }
                break;
            }
        }
    }
    None
}

fn parse_return<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    index: usize,
) -> Option<(Statement<'a>, usize)> {
    if index < tokens.len() {
        if expect_keyword(tokens, index, "return") {
            // Check if we have an empty return
            let i = index + 1;
            if expect_syntax(tokens, i, &SYNTAX_SC.to_string()) {
                return Some((Statement::Return(Return { exp: None }), i + 1));
            }
            if let Some((exp, i)) = parse_expression(raw, tokens, i, 0) {
                return Some((Statement::Return(Return { exp: Some(exp) }), 1));
            }
        }
    }
    None
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

// The functions below are concerned with parsing expressions.
fn try_parethesized<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    index: usize,
) -> Option<(Expression<'a>, usize)> {
    // No need to check for index < tokens.len(), it is being checked in the caller.
    if expect_syntax(tokens, index, &SYNTAX_OP.to_string()) {
        let exp_begin_idx = index + 1;
        let (exp, i) = parse_expression(raw, tokens, exp_begin_idx, 0)?;
        if expect_syntax(tokens, i, &SYNTAX_CP.to_string()) {
            return Some((exp, i + 1));
        }
    }
    None
}

fn try_pre_unary_operator<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    index: usize,
) -> Option<(Expression<'a>, usize)> {
    if expect_operator(tokens, index, OP_INC) || expect_operator(tokens, index, OP_DEC) {
        let (exp, idx) = parse_primary(raw, tokens, index + 1)?;
        return Some((
            Expression::UnaryOperation(UnaryOperation {
                is_pre: true,
                operator: tokens[index].clone(),
                exp: Box::new(exp),
            }),
            idx,
        ));
    }
    None
}

fn try_post_unary_operator<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    index: usize,
) -> Option<(Expression<'a>, usize)> {
    if let Some((exp, i)) = try_parethesized(raw, tokens, index) {
        if expect_operator(tokens, i, OP_INC) || expect_operator(tokens, i, OP_DEC) {
            return Some((
                Expression::UnaryOperation(UnaryOperation {
                    is_pre: false,
                    operator: tokens[i].clone(),
                    exp: Box::new(exp),
                }),
                i + 1,
            ));
        }
    }
    if tokens[index].kind() == TokenKind::Number || tokens[index].kind() == TokenKind::Identifier {
        let possible_unary_op = index + 1;
        if expect_operator(tokens, possible_unary_op, OP_INC)
            || expect_operator(tokens, possible_unary_op, OP_DEC)
        {
            let primary_after = possible_unary_op + 1;
            match tokens[index].kind() {
                TokenKind::Identifier => {
                    return Some((
                        Expression::UnaryOperation(UnaryOperation {
                            is_pre: false,
                            operator: tokens[possible_unary_op].clone(),
                            exp: Box::new(Expression::Literal(Literal::Identifier(
                                tokens[index].clone(),
                            ))),
                        }),
                        primary_after,
                    ));
                }
                TokenKind::Number => {
                    return Some((
                        Expression::UnaryOperation(UnaryOperation {
                            is_pre: false,
                            operator: tokens[possible_unary_op].clone(),
                            exp: Box::new(Expression::Literal(Literal::Number(
                                tokens[index].clone(),
                            ))),
                        }),
                        primary_after,
                    ));
                }
                _ => {}
            }
        }
    }
    None
}

fn try_function_call<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    index: usize,
) -> Option<(Expression<'a>, usize)> {
    if expect_identifier(tokens, index) {
        let op_parenthesis = index + 1;
        if expect_syntax(tokens, op_parenthesis, &SYNTAX_OP.to_string()) {
            let mut i = op_parenthesis + 1;
            let mut expressions: Vec<Expression<'a>> = Vec::new();
            while i < tokens.len() {
                if expect_syntax(tokens, i, &SYNTAX_CP.to_string()) {
                    break;
                }
                if expressions.len() > 0 && !expect_syntax(tokens, i, &SYNTAX_CM.to_string()) {
                    return None;
                }
                i = i + 1;
                if let Some((exp, end)) = parse_expression(raw, tokens, i, 0) {
                    expressions.push(exp);
                    i = end;
                } else {
                    return None;
                }
            }
            return Some((
                Expression::FunctionCall(FunctionCall {
                    name: tokens[index].clone(),
                    arguments: expressions,
                }),
                i + 1,
            ));
        }
    }
    None
}

fn try_identifier<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    index: usize,
) -> Option<(Expression<'a>, usize)> {
    if tokens[index].kind() == TokenKind::Identifier {
        return Some((
            Expression::Literal(Literal::Identifier(tokens[index].clone())),
            index + 1,
        ));
    }
    None
}

fn try_number<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    index: usize,
) -> Option<(Expression<'a>, usize)> {
    if tokens[index].kind() == TokenKind::Identifier {
        return Some((
            Expression::Literal(Literal::Number(tokens[index].clone())),
            index + 1,
        ));
    }
    None
}

fn parse_primary<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    index: usize,
) -> Option<(Expression<'a>, usize)> {
    if index < tokens.len() {
        let primary_expressions_parser = [
            try_pre_unary_operator,
            try_post_unary_operator,
            try_parethesized,
            try_function_call,
            try_identifier,
            try_number,
        ];
        for primary_parser in primary_expressions_parser {
            if let Some(duo) = primary_parser(raw, tokens, index) {
                return Some(duo);
            }
        }
    }
    None
}

fn parse_expression<'a>(
    raw: &[char],
    tokens: &[Token<'a>],
    index: usize,
    min_precedence: usize,
) -> Option<(Expression<'a>, usize)> {
    if index < tokens.len() {
        let (mut lhs, mut i) = parse_primary(raw, tokens, index)?;
        while i < tokens.len() {
            if tokens[i].kind() != TokenKind::Operator {
                break;
            }
            let new_precedence = operator_precedence(tokens[i].value())
                + if is_right_associative(tokens[i].value()) {
                    0
                } else {
                    1
                };
            if new_precedence < min_precedence {
                break;
            }
            let (rhs, new_i) = parse_expression(raw, tokens, i + 1, new_precedence)?;
            lhs = Expression::BinaryOperation(BinaryOperation {
                operator: tokens[i].clone(),
                left: Box::new(lhs),
                right: Box::new(rhs),
            });
            i = new_i;
        }

        return Some((lhs, i));
    }
    None
}
