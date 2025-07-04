use super::helpers::*;

#[derive(Copy, Clone, Debug)]
pub struct Location {
    col: i32,
    line: i32,
    index: usize,
}

impl Location {
    pub fn new() -> Self {
        Self {
            col: 0,
            line: 0,
            index: 0,
        }
    }

    pub fn increment(self: &Self, newline: bool) -> Self {
        if newline {
            Self {
                index: self.index + 1,
                col: 0,
                line: self.line + 1,
            }
        } else {
            Self {
                index: self.index + 1,
                col: self.col + 1,
                line: self.line,
            }
        }
    }

    pub fn increment_n(self: &Self, n: usize) -> Self {
        let mut other = *self;
        for i in 0..n {
            other = other.increment(false);
        }
        other
    }

    pub fn debug<S: Into<String>>(self: &Self, raw: &[char], msg: S) -> String {
        let mut line = 0;
        let mut line_str = String::new();

        for c in raw {
            if *c == '\n' {
                line += 1;
                if !line_str.is_empty() {
                    break;
                }
                continue;
            }
            if self.line == line {
                line_str.push_str(&c.to_string());
            }
        }

        let space = " ".repeat(self.col as usize);
        format!("{}\n\n{}\n{}^ Near here", msg.into(), line_str, space)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TokenKind {
    Identifier,
    Syntax,
    Keyword,
    Number,
    Operator,
}

#[derive(Clone)]
pub struct Token<'a> {
    value: &'a [char],
    kind: TokenKind,
    loc: Location,
}

impl<'a> Token<'a> {
    pub fn new(value: &'a [char], kind: TokenKind, loc: Location) -> Self {
        Self { value, kind, loc }
    }

    pub fn kind(self: &Self) -> TokenKind {
        self.kind
    }

    pub fn location(self: &Self) -> Location {
        self.loc
    }

    pub fn value(self: &Self) -> &[char] {
        self.value
    }
}

type LexerFn = fn(&'_ [char], Location) -> Option<(Token<'_>, Location)>;

pub struct Lexer {}

impl Lexer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn lex(lua: &'_ [char]) -> Result<Vec<Token<'_>>, String> {
        let mut loc = Location::new();
        let size = lua.len();
        let mut tokens: Vec<Token<'_>> = vec![];

        let lexers: [LexerFn; 5] = [
            lex_keyword,
            lex_identifier,
            lex_number,
            lex_operator,
            lex_syntax,
        ];

        'outer: while loc.index < size {
            loc = eat_whitespace(lua, loc);
            if loc.index == size {
                break;
            }
            for lexer in lexers {
                if let Some((token, location)) = lexer(lua, loc) {
                    loc = location;
                    tokens.push(token);
                    continue 'outer;
                }
            }

            return Err(loc.debug(lua, "Unexpected character while lexing."));
        }
        Ok(tokens)
    }
}

fn eat_whitespace(raw: &[char], initial_loc: Location) -> Location {
    let mut next_loc = initial_loc;
    while let Some(size) = is_whitespace(&raw[next_loc.index..]) {
        if raw[next_loc.index] == '\n' {
            next_loc = next_loc.increment(true);
        } else {
            next_loc = next_loc.increment_n(size);
        }
    }
    next_loc
}

fn lex_number(raw: &'_ [char], initial_loc: Location) -> Option<(Token<'_>, Location)> {
    if let Some(size) = is_number(&raw[initial_loc.index..]) {
        let next_loc = initial_loc.increment_n(size);
        if is_operator(&raw[next_loc.index..]).is_some()
            || is_syntax(&raw[next_loc.index..]).is_some()
            || is_whitespace(&raw[next_loc.index..]).is_some()
        {
            return Some((
                Token::new(
                    &raw[initial_loc.index..next_loc.index],
                    TokenKind::Number,
                    initial_loc,
                ),
                next_loc,
            ));
        }
    }
    None
}

fn lex_identifier(raw: &'_ [char], initial_loc: Location) -> Option<(Token<'_>, Location)> {
    if let Some(size) = is_identifier(&raw[initial_loc.index..]) {
        let next_loc = initial_loc.increment_n(size);
        return Some((
            Token::new(
                &raw[initial_loc.index..next_loc.index],
                TokenKind::Identifier,
                initial_loc,
            ),
            next_loc,
        ));
    }
    None
}

fn lex_keyword(raw: &'_ [char], initial_loc: Location) -> Option<(Token<'_>, Location)> {
    if let Some(size) = is_keyword(&raw[initial_loc.index..]) {
        let next_loc = initial_loc.increment_n(size);
        if is_whitespace(&raw[next_loc.index..]).is_some()
            || is_syntax(&raw[next_loc.index..]).is_some()
        {
            return Some((
                Token::new(
                    &raw[initial_loc.index..next_loc.index],
                    TokenKind::Keyword,
                    initial_loc,
                ),
                next_loc,
            ));
        }
    }
    None
}

fn lex_syntax(raw: &'_ [char], initial_loc: Location) -> Option<(Token<'_>, Location)> {
    if let Some(size) = is_syntax(&raw[initial_loc.index..]) {
        let next_loc = initial_loc.increment_n(size);
        return Some((
            Token::new(
                &raw[initial_loc.index..next_loc.index],
                TokenKind::Syntax,
                initial_loc,
            ),
            next_loc,
        ));
    }
    None
}

fn lex_operator(raw: &'_ [char], initial_loc: Location) -> Option<(Token<'_>, Location)> {
    match is_operator(&raw[initial_loc.index..]) {
        Some(size) => {
            let next_loc = initial_loc.increment_n(size);
            return Some((
                Token::new(
                    &raw[initial_loc.index..next_loc.index],
                    TokenKind::Operator,
                    initial_loc,
                ),
                next_loc,
            ));
        }
        None => return None,
    }
}
