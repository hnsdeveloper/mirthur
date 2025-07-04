// Whitespace
pub const SPACE: &str = " ";
pub const NEW_LINE: &str = "\n";
pub const CARRIAGE_RETURN: &str = "\r";
pub const TAB: &str = "\t";

pub const WHITESPACE: [&str; 4] = [SPACE, NEW_LINE, CARRIAGE_RETURN, TAB];

// Keywords
pub const KW_FUNCTION: &str = "function";
pub const KW_END: &str = "end";
pub const KW_IF: &str = "if";
pub const KW_ELSE: &str = "else";
pub const KW_ELIF: &str = "elif";
pub const KW_THEN: &str = "then";
pub const KW_LET: &str = "let";
pub const KW_RETURN: &str = "return";

pub const KEYWORDS: [&str; 8] = [
    KW_FUNCTION,
    KW_END,
    KW_IF,
    KW_ELSE,
    KW_ELIF,
    KW_THEN,
    KW_LET,
    KW_RETURN,
];

// Syntax
pub const SYNTAX_SC: char = ';';
pub const SYNTAX_EQ: char = '=';
pub const SYNTAX_OP: char = '(';
pub const SYNTAX_CP: char = ')';
pub const SYNTAX_CM: char = ',';

pub const SYNTAX: [char; 5] = [SYNTAX_SC, SYNTAX_EQ, SYNTAX_OP, SYNTAX_CP, SYNTAX_CM];

// Operators
pub const OP_PLUS: &str = "+";
pub const OP_MINUS: &str = "-";
pub const OP_POW: &str = "**";
pub const OP_MUL: &str = "*";
pub const OP_DIV: &str = "/";
pub const OP_MOD: &str = "%";
pub const OP_SHL: &str = "<<";
pub const OP_LT: &str = "<";
pub const OP_SHR: &str = ">>";
pub const OP_GT: &str = ">";
pub const OP_CMP: &str = "==";
pub const OP_DOT: &str = ".";

pub const OPERATORS: [&str; 12] = [
    OP_PLUS, OP_MINUS, OP_POW, OP_MUL, OP_DIV, OP_MOD, OP_SHL, OP_LT, OP_SHR, OP_GT, OP_CMP, OP_DOT,
];
