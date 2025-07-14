use std::collections::{HashMap, HashSet};
use strum::IntoEnumIterator;

use super::constants::*;
use super::cpu::{Instruction, Reg};
use super::lexer::Token;
use super::parser::*;

pub type Instructions = Vec<u8>;

enum SymbolType {
    Function(u64),
    Global(String, u64),
}

enum LocalLocation {
    // A positive offset from the current stack pointer.
    Stack(u64),
    Register(Reg),
}

type SymbolTable = HashMap<String, SymbolType>;

pub struct Compiler<'a> {
    instr: Instructions,
    ast: AST<'a>,
}

impl<'a> Compiler<'a> {
    pub fn build(ast: AST<'a>) -> Self {
        Self {
            instr: Vec::new(),
            ast,
        }
    }

    fn adjust_function_call(
        self: &mut Self,
        locations: Vec<usize>,
        adjusted_symbol_table: &HashMap<u64, u64>,
    ) {
        for location in locations {
            // Unwrapping should be safe given that the instructions are emmited by the compiler itself
            let (ins, _) = Instruction::deserialize(&self.instr[location..]).unwrap();
            if let Instruction::Call(addr) = ins {
                let ins = Instruction::Call(*adjusted_symbol_table.get(&addr).unwrap());
                let serialized = ins.serialize();
                for j in 0..serialized.len() {
                    self.instr[location + j] = serialized[j];
                }
            }
        }
    }

    fn build_global_symbol_table(ast: &AST) -> Result<SymbolTable, String> {
        let mut symbols = HashMap::new();

        for stmt in ast {
            match stmt {
                Statement::FunctionDeclaration(function_declaration) => {
                    let name = String::from_iter(function_declaration.name.value());
                }
                // Later, support for global variables should be added.
                _ => {
                    return Err("Invalid statement at global scope.".into());
                }
            }
        }

        Ok(symbols)
    }

    fn build_registers_set() -> HashSet<Reg> {
        Reg::general_purpose_registers().into_iter().collect()
    }

    fn build_local_locations(
        parameters: &Vec<Token>,
        registers: &mut HashSet<Reg>,
    ) -> Result<HashMap<String, LocalLocation>, String> {
        let mut locations = HashMap::new();
        for (i, reg) in Reg::argument_registers().iter().enumerate() {
            if i < parameters.len() {
                let symbol = String::from_iter(parameters[i].value());
                if let Some(_) = locations.insert(symbol.clone(), LocalLocation::Register((*reg))) {
                    return Err(format!(
                        "Parameter symbol can't show more than once : {}",
                        symbol
                    ));
                }
                registers.remove(reg);
            }
        }

        if Reg::argument_registers().len() < parameters.len() {
            let slc = &parameters[Reg::argument_registers().len()..];
            for (i, p) in slc.iter().enumerate().rev() {
                let symbol = String::from_iter(p.value());
                // TODO: We calculate as such given that for now we only deal with 64 bit integers
                let j = slc.len() - i;
                if let Some(_) = locations.insert(
                    symbol.clone(),
                    LocalLocation::Stack((j * size_of::<u64>()) as u64),
                ) {
                    return Err(format!(
                        "Parameter symbol can't show more than once : {}",
                        symbol
                    ));
                }
            }
        }

        Ok(locations)
    }

    fn emit_op(instructions: &mut Instructions, operator: &[char], lhs: Reg, rhs: Reg) {
        let op = String::from_iter(operator);
        let ins;
        match op.as_str() {
            OP_PLUS => {
                ins = Instruction::Add(lhs, rhs);
            }
            OP_MINUS => {
                ins = Instruction::Sub(lhs, rhs);
            }
            OP_MUL => {
                ins = Instruction::Mul(lhs, rhs);
            }
            OP_DIV => {
                ins = Instruction::Div(lhs, rhs);
            }
            OP_POW => {
                ins = Instruction::Pow(lhs, rhs);
            }
            _ => unreachable!(),
        }

        instructions.extend(ins.serialize());
    }

    fn emit_push_reg(instructions: &mut Instructions, reg: Reg) {
        instructions.extend_from_slice(&Instruction::PushReg(reg).serialize());
    }

    fn emit_write_to_reg(instructions: &mut Instructions, r_what: Reg, r_where: Reg, offset: u64) {
        instructions.extend_from_slice(&Instruction::StoreReg(r_what, r_where, offset).serialize());
    }

    fn emit_load_reg(instructions: &mut Instructions, r_target: Reg, r_where: Reg, offset: u64) {
        instructions
            .extend_from_slice(&Instruction::LoadReg(r_target, r_where, offset).serialize());
    }

    fn emit_mov_imm(instructions: &mut Instructions, r_target: Reg, val: u64) {
        instructions.extend_from_slice(&Instruction::MovImm(r_target, val).serialize());
    }

    fn emit_pop_reg(instructions: &mut Instructions, r_target: Reg) {
        instructions.extend_from_slice(&Instruction::PopReg(r_target).serialize());
    }

    fn find_register_local_name(
        reg: Reg,
        locals: &HashMap<String, LocalLocation>,
    ) -> Option<String> {
        locals
            .iter()
            .filter_map(|(s, loc)| {
                if let LocalLocation::Register(r) = loc {
                    if *r == reg {
                        return Some(s.clone());
                    }
                }
                None
            })
            .last()
    }

    fn spill_local_to_stack(
        name: &String,
        instructions: &mut Instructions,
        locals: &mut HashMap<String, LocalLocation>,
        stack_free_slots: &mut Vec<u64>,
        registers: &mut HashSet<Reg>,
    ) {
        let (_, r) = match locals.get_key_value(name) {
            Some((n, LocalLocation::Register(r))) => (n.clone(), *r),
            _ => panic!("Invalid call to spill_local_to_stack."),
        };
        let offset: u64;
        if let Some(v) = stack_free_slots.pop() {
            offset = v;
            Self::emit_write_to_reg(instructions, r, Reg::SP, offset);
        } else {
            offset = 0;
            Self::emit_push_reg(instructions, r);
            for (_, loc) in locals.iter_mut() {
                match loc {
                    LocalLocation::Stack(offset) => {
                        *loc = LocalLocation::Stack(*offset + size_of::<u64>() as u64);
                    }
                    _ => {}
                }
            }
        }
        *locals.get_mut(name).unwrap() = LocalLocation::Stack(offset);
        registers.insert(r);
    }

    fn load_local_from_stack(
        name: &String,
        instructions: &mut Instructions,
        registers: &mut HashSet<Reg>,
        locals: &mut HashMap<String, LocalLocation>,
        stack_free_slots: &mut Vec<u64>,
    ) -> Reg {
        if registers.len() == 0 {
            let (name, _) = locals.iter().last().unwrap();
            let name = name.clone();
            Self::spill_local_to_stack(&name, instructions, locals, stack_free_slots, registers);
        }
        let r = *registers.iter().last().unwrap();
        registers.remove(&r);
        if let Some(LocalLocation::Stack(offset)) = locals.get_mut(name) {
            if *offset == 0 {
                Self::emit_pop_reg(instructions, r);
                for (_, loc) in locals.iter_mut() {
                    match loc {
                        LocalLocation::Stack(of) => {
                            *loc = LocalLocation::Stack(*of - size_of::<u64>() as u64);
                        }
                        _ => {}
                    }
                }
            } else {
                Self::emit_load_reg(instructions, r, Reg::SP, *offset);
                stack_free_slots.push(*offset);
            }
        }
        *locals.get_mut(name).unwrap() = LocalLocation::Register(r);
        r
    }

    fn compile_identifier(
        name: &[char],
        instructions: &mut Instructions,
        registers: &mut HashSet<Reg>,
        locals: &mut HashMap<String, LocalLocation>,
        stack_free_slots: &mut Vec<u64>,
    ) -> Result<Reg, String> {
        let name = String::from_iter(name);
        if let Some(loc) = locals.get(&name) {
            match loc {
                LocalLocation::Stack(_) => {
                    return Ok(Self::load_local_from_stack(
                        &name,
                        instructions,
                        registers,
                        locals,
                        stack_free_slots,
                    ));
                }
                LocalLocation::Register(reg) => {
                    return Ok(*reg);
                }
            }
        } else {
            return Err(format!(
                "Symbol {} not in scope while compiling expression.",
                name
            ));
        }
    }

    fn compile_number(
        v: &[char],
        instructions: &mut Instructions,
        registers: &mut HashSet<Reg>,
        locals: &mut HashMap<String, LocalLocation>,
        stack_free_slots: &mut Vec<u64>,
    ) -> Result<Reg, String> {
        let t = String::from_iter(v);
        // This should probably be moved to the parsing step
        let v = if let Ok(v) = t.parse::<u64>() {
            v
        } else {
            return Err(format!("Invalid number as value. Value was {}", t));
        };
        if registers.len() == 0 {
            let (name, _) = locals.iter().last().unwrap();
            let name = name.clone();
            Self::spill_local_to_stack(&name, instructions, locals, stack_free_slots, registers);
        }
        let r = *registers.iter().last().unwrap();
        registers.remove(&r);
        Self::emit_mov_imm(instructions, r, v);
        Ok(r)
    }

    fn compile_expression(
        instructions: &mut Instructions,
        expression: &Expression,
        provisory_symbol_table: &SymbolTable,
        function_call_adjustments: &mut Vec<usize>,
        registers: &mut HashSet<Reg>,
        locals: &mut HashMap<String, LocalLocation>,
        stack_free_slots: &mut Vec<u64>,
    ) -> Result<Reg, String> {
        match expression {
            Expression::FunctionCall(function_call) => {}
            Expression::UnaryOperation(unary_operation) => todo!(),
            Expression::BinaryOperation(binary_operation) => {
                let lhs = Self::compile_expression(
                    instructions,
                    &binary_operation.left,
                    provisory_symbol_table,
                    function_call_adjustments,
                    registers,
                    locals,
                    stack_free_slots,
                )?;
                // Given that we modify lhs, if lhs is a local, we want to keep the original value as is
                // So we split it
                if let Some(local_name) = Self::find_register_local_name(lhs, locals) {
                    // It will free up the register
                    Self::spill_local_to_stack(
                        &local_name,
                        instructions,
                        locals,
                        stack_free_slots,
                        registers,
                    );
                    // As it is not what we want
                    registers.remove(&lhs);
                }
                let rhs = Self::compile_expression(
                    instructions,
                    &binary_operation.right,
                    provisory_symbol_table,
                    function_call_adjustments,
                    registers,
                    locals,
                    stack_free_slots,
                )?;

                Self::emit_op(instructions, binary_operation.operator.value(), lhs, rhs);
                if let None = Self::find_register_local_name(rhs, locals) {
                    registers.insert(rhs);
                }
                return Ok(lhs);
            }
            Expression::Literal(literal) => match literal {
                Literal::Identifier(token) => {
                    return Self::compile_identifier(
                        token.value(),
                        instructions,
                        registers,
                        locals,
                        stack_free_slots,
                    );
                }
                Literal::Number(token) => {
                    return Self::compile_number(
                        token.value(),
                        instructions,
                        registers,
                        locals,
                        stack_free_slots,
                    );
                }
            },
        }
        todo!()
    }

    fn get_provisory_function_symbol_location(
        name: &[char],
        provisory_symbol_table: &SymbolTable,
    ) -> Result<u64, String> {
        let name = String::from_iter(name);
        let provisory_symbol_location =
            if let SymbolType::Function(v) = provisory_symbol_table.get(&name).unwrap() {
                *v
            } else {
                return Err(format!("Symbol {} is not function symbol.", name));
            };
        Ok(provisory_symbol_location)
    }

    fn compile_function_declaration(
        fd: &FunctionDeclaration,
        instructions: &mut Instructions,
        provisory_symbol_table: &SymbolTable,
        adjusted_table: &mut HashMap<u64, u64>,
        function_call_adjustments: &mut Vec<usize>,
    ) -> Result<(), String> {
        let prov_fn_loc =
            Self::get_provisory_function_symbol_location(fd.name.value(), provisory_symbol_table)?;
        adjusted_table.insert(prov_fn_loc, instructions.len() as u64);
        let mut registers = Self::build_registers_set();
        let mut locals = Self::build_local_locations(&fd.parameters, &mut registers)?;
        let mut stack_free_slots: Vec<u64> = Vec::new();
        for stmt in &fd.body {
            match stmt {
                Statement::Expression(expression) => {
                    let r = Self::compile_expression(
                        instructions,
                        expression,
                        provisory_symbol_table,
                        function_call_adjustments,
                        &mut registers,
                        &mut locals,
                        &mut stack_free_slots,
                    )?;
                    //if !Self::find_register_local_name(r, &mut locals) {
                    //    registers.insert(r);
                    //}
                }
                Statement::If(_) => todo!(),
                Statement::Return(_) => todo!(),
                Statement::Let(_) => todo!(),
                _ => {
                    return Err("Statement not allowed inside function declaration.".into());
                }
            }
        }
        todo!()
    }

    pub fn compile(self: &mut Self) -> Result<Instructions, String> {
        // First we populate the symbol table
        let provisory_symbol_table = Self::build_global_symbol_table(&self.ast)?;
        let mut adjustment_table: HashMap<u64, u64> = HashMap::new();
        let mut instructions_to_adjust: Vec<usize> = Vec::new();
        for stmt in &self.ast {
            match stmt {
                Statement::FunctionDeclaration(function_declaration) => {
                    Self::compile_function_declaration(
                        function_declaration,
                        &mut self.instr,
                        &provisory_symbol_table,
                        &mut adjustment_table,
                        &mut instructions_to_adjust,
                    )?;
                }
                _ => {
                    return Err(
                        "Only function declarations supported at the top level for now.".into(),
                    );
                }
            }
        }

        todo!()
    }
}
