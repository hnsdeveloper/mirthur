use super::constants::*;
use super::cpu::{Instruction, Reg};
use super::parser::*;
use ctx::{CompilerContext, LocalInfo, LocalLocation};

pub type Instructions = Vec<u8>;

mod ctx {
    use super::super::cpu::Reg;
    use super::super::parser::*;
    use std::collections::{HashMap, HashSet};

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum LocalLocation {
        // A positive offset from the current stack pointer.
        Stack(u64),
        Register(Reg),
    }

    #[derive(Clone)]
    pub struct LocalInfo {
        name: String,
        location: LocalLocation,
        scope: usize,
    }

    impl LocalInfo {
        fn new(name: String, loc: LocalLocation, scope: usize) -> Self {
            Self {
                name,
                location: loc,
                scope,
            }
        }

        pub fn get_name(&self) -> &String {
            &self.name
        }

        pub fn get_location(&self) -> &LocalLocation {
            &self.location
        }

        pub fn get_scope(&self) -> usize {
            self.scope
        }

        pub fn get_register(&self) -> Option<Reg> {
            match self.location {
                LocalLocation::Register(r) => Some(r),
                LocalLocation::Stack(_) => None,
            }
        }

        pub fn get_stack_offset(&self) -> Option<u64> {
            match self.location {
                LocalLocation::Register(_) => None,
                LocalLocation::Stack(v) => Some(v),
            }
        }
    }

    pub struct CompilerContext {
        instructions: Vec<u8>,
        free_registers: HashSet<Reg>,
        function_temp_location_counter: u64,
        function_provisory_symbol_table: HashMap<String, u64>,
        function_adjusted_symbol_table: HashMap<u64, u64>,
        scoped_locals: Vec<HashMap<String, LocalLocation>>,
        stack_free_slots: HashSet<u64>,
        temp_name_counter: usize,
    }

    impl CompilerContext {
        pub fn new() -> Self {
            todo!()
        }

        pub fn extend_instructions(&mut self, data: &[u8]) {
            self.instructions.extend_from_slice(data);
        }

        pub fn build_function_temporary_location(&mut self, ast: &AST) -> Result<(), String> {
            for stmt in ast {
                if let Statement::FunctionDeclaration(fd) = stmt {
                    let name = String::from_iter(fd.name.value());
                    self.set_function_temporary_location(name)?;
                }
            }
            Ok(())
        }

        pub fn get_provisory_function_symbol_location(
            &mut self,
            name: String,
        ) -> Result<u64, String> {
            if let Some(v) = self.function_provisory_symbol_table.get(&name) {
                return Ok(*v);
            }
            Err(format!(
                "Symbol {} provisory location not registered.",
                name
            ))
        }

        fn set_function_temporary_location(&mut self, name: String) -> Result<u64, String> {
            let n = name.clone();
            if self
                .function_provisory_symbol_table
                .insert(name, self.function_temp_location_counter)
                .is_some()
            {
                return Err(format!("Function symbol already defined : {}", n));
            }
            let v = self.function_temp_location_counter;
            self.function_temp_location_counter += 1;
            Ok(v)
        }

        fn set_function_fixed_location_curr_instruction(
            &mut self,
            temp_location: u64,
        ) -> Result<u64, String> {
            let s = self.instructions.len() as u64;
            if self
                .function_adjusted_symbol_table
                .insert(temp_location, s)
                .is_some()
            {
                return Err(format!("Symbol at {} already fixated.", temp_location));
            }
            Ok(s)
        }

        // Makes a certain register available
        pub fn deallocate_register(&mut self, reg: Reg) {
            self.free_registers.insert(reg);
        }

        // Takes the first register it finds free.
        pub fn allocate_register(&mut self) -> Option<Reg> {
            if let Some(r) = self.free_registers.iter().last() {
                let r = *r;
                self.free_registers.remove(&r);
                return Some(r);
            }
            None
        }

        pub fn get_symbol_in_register(
            &mut self,
            avoid_symbols: Option<Vec<&LocalInfo>>,
        ) -> Option<LocalInfo> {
            let avoid = avoid_symbols.unwrap_or_default();
            for (scope, symbols) in self.scoped_locals.iter_mut().rev().enumerate() {
                for (symbol_name, location) in symbols {
                    if let LocalLocation::Register(_) = *location {
                        for avoid_sym in avoid.iter() {
                            if symbol_name == avoid_sym.get_name() && scope == avoid_sym.get_scope()
                            {
                                continue;
                            }
                        }
                        return Some(LocalInfo::new(symbol_name.clone(), *location, scope));
                    }
                }
            }
            None
        }

        pub fn create_symbol(
            &mut self,
            name: String,
            location: LocalLocation,
        ) -> Result<LocalInfo, String> {
            for symbol_table in &self.scoped_locals {
                for (s, loc) in symbol_table {
                    if *loc == location {
                        return Err(format!(
                            "Can't create symbol {} in symbol {} location.",
                            name, s
                        ));
                    }
                }
            }
            let i = self.scoped_locals.len() - 1;
            let symbol_table = &mut self.scoped_locals[i];
            // We don't have to create the symbol again, just utilize the old space and done.
            if let Some(l) = symbol_table.get(&name) {
                return Ok(LocalInfo::new(name, *l, i));
            }
            match location {
                LocalLocation::Register(r) => {
                    if !self.free_registers.contains(&r) {
                        return Err(
                            "Can't create symbol on register. Register already in use.".into()
                        );
                    }
                }
                LocalLocation::Stack(offset) => {
                    if !self.stack_free_slots.contains(&offset) {
                        return Err("Can't create symbol on stack. Space already in use.".into());
                    }
                }
            }

            symbol_table.insert(name.clone(), location);
            Ok(LocalInfo::new(name, location, i))
        }

        pub fn move_symbol(
            &mut self,
            symbol: &LocalInfo,
            new_location: LocalLocation,
        ) -> Result<LocalInfo, String> {
            if symbol.get_scope() >= self.scoped_locals.len() {
                return Err(format!("Inexistent scope for symbol {}", symbol.get_name()));
            }
            let symbols = &mut self.scoped_locals[symbol.get_scope()];
            if !symbols.contains_key(symbol.get_name()) {
                return Err(format!(
                    "Symbol {} inexistent on scope {}",
                    symbol.get_name(),
                    symbol.get_scope()
                ));
            }
            let location = symbols.get_mut(symbol.get_name()).unwrap();
            if *location == new_location {
                return Err(
                    "Attempting to move symbol to location which it already is present.".into(),
                );
            }
            // Check if the new location is available and removes it from the available pool
            match new_location {
                LocalLocation::Register(r) => {
                    if !self.free_registers.contains(&r) {
                        // TODO: pretty print which register
                        return Err("Register in use. Can't assing local to register.".into());
                    }
                    self.free_registers.remove(&r);
                }
                LocalLocation::Stack(offset) => {
                    if !self.stack_free_slots.contains(&offset) {
                        return Err(format!("Can't assing symbol to offset {}.", offset));
                    }
                    self.stack_free_slots.remove(&offset);
                }
            };
            // Free the old location
            match location {
                LocalLocation::Register(r) => self.free_registers.insert(*r),
                LocalLocation::Stack(offset) => self.stack_free_slots.insert(*offset),
            };
            *location = new_location;
            Ok(LocalInfo::new(
                symbol.get_name().clone(),
                *location,
                symbol.get_scope(),
            ))
        }

        pub fn destroy_symbol(&mut self, symbol: &LocalInfo) -> Result<(), String> {
            if symbol.get_scope() >= self.scoped_locals.len() {
                return Err("Scope non existent.".into());
            }
            let i = symbol.get_scope();
            let symbol_table = &mut self.scoped_locals[i];
            if let Some(l) = symbol_table.remove(symbol.get_name()) {
                match l {
                    LocalLocation::Register(r) => self.free_registers.insert(r),
                    LocalLocation::Stack(s) => self.stack_free_slots.insert(s),
                };
                Ok(())
            } else {
                Err(format!(
                    "Attempting to destroy non existent symbol {}",
                    symbol.get_name()
                ))
            }
        }

        pub fn find_local_info_by_name(&mut self, name: &String) -> Option<LocalInfo> {
            for (scope, symbols) in self.scoped_locals.iter().enumerate().rev() {
                if let Some(location) = symbols.get(name) {
                    return Some(LocalInfo::new(name.clone(), *location, scope));
                }
            }
            None
        }

        pub fn find_local_info_by_location(
            &mut self,
            location: LocalLocation,
        ) -> Option<LocalInfo> {
            for (scope, symbols) in self.scoped_locals.iter().enumerate().rev() {
                let e = symbols.iter().find(|(_, l)| {
                    if **l == location {
                        return true;
                    }
                    false
                });
                if let Some((s, l)) = e {
                    return Some(LocalInfo::new(s.clone(), *l, scope));
                }
            }
            None
        }

        pub fn get_temporary_name(&mut self) -> String {
            self.temp_name_counter += 1;
            let i = self.temp_name_counter;
            i.to_string()
        }

        pub fn is_temporary(&mut self, name: &str) -> bool {
            name.parse::<usize>().is_ok()
        }

        pub fn push_stack(&mut self) {
            for symbol_table in self.scoped_locals.iter_mut() {
                for (_, symbol_loc) in symbol_table.iter_mut() {
                    if let LocalLocation::Stack(offset) = symbol_loc {
                        let v = *offset + size_of::<u64>() as u64;
                        *symbol_loc = LocalLocation::Stack(v);
                    }
                }
            }
            let p = self
                .stack_free_slots
                .clone()
                .into_iter()
                .map(|v| v + size_of::<u64>() as u64);
            self.stack_free_slots = p.collect();
            self.stack_free_slots.insert(0);
        }

        pub fn pop_stack(&mut self) -> Result<(), String> {
            for symbol_table in self.scoped_locals.iter_mut() {
                for (_, symbol_loc) in symbol_table.iter_mut() {
                    if let LocalLocation::Stack(offset) = symbol_loc {
                        if *offset == 0 {
                            return Err(
                                "Attempting to destroy symbol through stack popping.".into()
                            );
                        }
                        let v = *offset - size_of::<u64>() as u64;
                        *symbol_loc = LocalLocation::Stack(v);
                    }
                }
            }
            let p = self.stack_free_slots.clone().into_iter().filter_map(|v| {
                if v != 0 {
                    return Some(v - size_of::<u64>() as u64);
                }
                None
            });
            self.stack_free_slots = p.collect();
            Ok(())
        }

        pub fn get_stack_free_slot(&self) -> Option<&u64> {
            self.stack_free_slots.iter().last()
        }

        pub fn enter_function_declaration(&mut self, fd: &FunctionDeclaration) {
            let name = String::from_iter(fd.name.value());
            let tmp = *self.function_provisory_symbol_table.get(&name).unwrap();
            let _ = self.set_function_fixed_location_curr_instruction(tmp);
            self.enter_scope();
            let parameters = fd
                .parameters
                .iter()
                .map(|x| String::from_iter(x.value()))
                .collect::<Vec<String>>();

            for (i, reg) in Reg::argument_registers().iter().enumerate() {
                if i < parameters.len() {
                    let _ =
                        self.create_symbol(parameters[i].clone(), LocalLocation::Register(*reg));
                }
            }
            if Reg::argument_registers().len() < parameters.len() {
                let slc = &parameters[Reg::argument_registers().len()..];
                for item in slc {
                    self.push_stack();
                    let offset = *self.get_stack_free_slot().unwrap();
                    let _ = self.create_symbol(item.clone(), LocalLocation::Stack(offset));
                }
            }
        }

        pub fn exit_function_declaration(&mut self) {
            self.exit_scope();
            for reg in Reg::general_purpose_registers() {
                self.deallocate_register(reg);
            }
        }

        pub fn enter_scope(&mut self) {
            self.scoped_locals.push(HashMap::new());
        }

        pub fn exit_scope(&mut self) {
            self.scoped_locals.pop();
        }
    }
}

pub struct Compiler<'a> {
    ast: AST<'a>,
}

impl<'a> Compiler<'a> {
    pub fn build(ast: AST<'a>) -> Self {
        Self { ast }
    }

    fn emit_op(ctx: &mut CompilerContext, operator: &[char], lhs: Reg, rhs: Reg) {
        let op = String::from_iter(operator);
        let ins = match op.as_str() {
            OP_PLUS => Instruction::Add(lhs, rhs),
            OP_MINUS => Instruction::Sub(lhs, rhs),
            OP_MUL => Instruction::Mul(lhs, rhs),
            OP_DIV => Instruction::Div(lhs, rhs),
            OP_POW => Instruction::Pow(lhs, rhs),
            _ => unreachable!(),
        };
        ctx.extend_instructions(&ins.serialize());
    }

    fn emit_load_reg_reg(ctx: &mut CompilerContext, r_target: Reg, r_where: Reg, offset: u64) {
        ctx.extend_instructions(&Instruction::LoadReg(r_target, r_where, offset).serialize());
    }

    fn emit_write_reg_reg(ctx: &mut CompilerContext, r_what: Reg, r_where: Reg, offset: u64) {
        ctx.extend_instructions(&Instruction::StoreReg(r_what, r_where, offset).serialize());
    }

    fn emit_mov_imm(ctx: &mut CompilerContext, r_target: Reg, val: u64) {
        ctx.extend_instructions(&Instruction::MovImm(r_target, val).serialize());
    }

    fn emit_mov_reg(ctx: &mut CompilerContext, r_target: Reg, r_source: Reg) {
        ctx.extend_instructions(&Instruction::Mov(r_target, r_source).serialize());
    }

    fn emit_push_reg(ctx: &mut CompilerContext, r: Reg) {
        ctx.extend_instructions(&Instruction::PushReg(r).serialize());
    }

    fn compile_literal(literal: &Literal, ctx: &mut CompilerContext) -> Result<LocalInfo, String> {
        match literal {
            Literal::Identifier(token) => {
                let name = String::from_iter(token.value());
                if let Some(info) = ctx.find_local_info_by_name(&name) {
                    return Ok(info);
                }
                Err(format!("Symbol {} not available on current scope.", name))
            }
            Literal::Number(token) => {
                let v = String::from_iter(token.value()).parse::<u64>().unwrap();
                let reg = if let Some(r) = ctx.allocate_register() {
                    r
                } else {
                    let s = ctx.get_symbol_in_register(None).unwrap();
                    if let Some(off) = ctx.get_stack_free_slot() {
                        let off = *off;
                        ctx.move_symbol(&s, LocalLocation::Stack(off))?;
                        Self::emit_write_reg_reg(ctx, s.get_register().unwrap(), Reg::SP, off);
                    } else {
                        ctx.push_stack();
                        let off = *ctx.get_stack_free_slot().unwrap();
                        ctx.move_symbol(&s, LocalLocation::Stack(off))?;
                        Self::emit_push_reg(ctx, s.get_register().unwrap());
                    }
                    s.get_register().unwrap()
                };
                let temp_name = ctx.get_temporary_name();
                Self::emit_mov_imm(ctx, reg, v);
                ctx.create_symbol(temp_name, LocalLocation::Register(reg))
            }
        }
    }

    fn bring_to_reg(
        ctx: &mut CompilerContext,
        symbol: &LocalInfo,
        leave_in_reg: Vec<&LocalInfo>,
    ) -> LocalInfo {
        // Not waste time computing the remaining, so we can simplify the code down the line.
        if symbol.get_register().is_some() {
            return symbol.clone();
        }
        if let Some(reg) = ctx.allocate_register() {
            let p = ctx
                .move_symbol(symbol, LocalLocation::Register(reg))
                .unwrap();
            Self::emit_load_reg_reg(ctx, reg, Reg::SP, symbol.get_stack_offset().unwrap());
            return p;
        }
        let purge = ctx.get_symbol_in_register(Some(leave_in_reg)).unwrap();
        if let Some(offset) = ctx.get_stack_free_slot() {
            let offset = *offset;
            let _ = ctx
                .move_symbol(&purge, LocalLocation::Stack(offset))
                .unwrap();
            Self::emit_write_reg_reg(ctx, purge.get_register().unwrap(), Reg::SP, offset);
        } else {
            ctx.push_stack();
            let offset = *ctx.get_stack_free_slot().unwrap();
            let _ = ctx.move_symbol(&purge, LocalLocation::Stack(offset));
            Self::emit_write_reg_reg(ctx, purge.get_register().unwrap(), Reg::SP, offset);
        }
        let reg = ctx.allocate_register().unwrap();
        let p = ctx
            .move_symbol(symbol, LocalLocation::Register(reg))
            .unwrap();
        Self::emit_load_reg_reg(ctx, reg, Reg::SP, symbol.get_stack_offset().unwrap());
        p
    }

    fn move_to_stack(symbol: &LocalInfo, ctx: &mut CompilerContext) -> LocalInfo {
        if symbol.get_stack_offset().is_some() {
            return symbol.clone();
        }
        if let Some(offset) = ctx.get_stack_free_slot() {
            let offset = *offset;
            let info = ctx
                .move_symbol(symbol, LocalLocation::Stack(offset))
                .unwrap();
            Self::emit_write_reg_reg(ctx, symbol.get_register().unwrap(), Reg::SP, offset);
            info
        } else {
            ctx.push_stack();
            let offset = *ctx.get_stack_free_slot().unwrap();
            let info = ctx
                .move_symbol(symbol, LocalLocation::Stack(offset))
                .unwrap();
            Self::emit_push_reg(ctx, symbol.get_register().unwrap());
            info
        }
    }

    fn adjust_for_binary_exp(
        left: &LocalInfo,
        right: &LocalInfo,
        ctx: &mut CompilerContext,
    ) -> (LocalInfo, LocalInfo) {
        // Lets enumerate the possibilities:
        // left is a temporary, on the stack, different than right
        // left is a temporary, on a register, different than right
        // left is a local, on a register, same as right
        // left is a local, on a register, different than right
        // left is a local, on the stack, different than right
        // left is a local, on the stack, same as right

        // After sorting this out, we will have to handle two different cases for the right side:
        // right is on a register
        // right is on the stack

        // Lets handle the temporary case first
        if ctx.is_temporary(left.get_name()) {
            if let LocalLocation::Register(_) = *left.get_location() {
                return (left.clone(), Self::bring_to_reg(ctx, right, vec![left]));
            } else {
                let a = Self::bring_to_reg(ctx, left, vec![right]);
                let b = Self::bring_to_reg(ctx, right, vec![&a]);
                return (a, b);
            }
        }

        match left.get_location() {
            LocalLocation::Register(_) => {
                if left.get_name() != right.get_name() {
                    let mut n_left = left.clone();
                    let r = if let Some(r) = ctx.allocate_register() {
                        r
                    } else {
                        n_left = Self::move_to_stack(left, ctx);
                        // The register that will become free will be the same we used, and thus the value will be there already
                        ctx.allocate_register().unwrap()
                    };
                    let temp_name = ctx.get_temporary_name();
                    let l = ctx
                        .create_symbol(temp_name, LocalLocation::Register(r))
                        .unwrap();
                    // If we still have left in a register, we have to copy its value to the new register
                    if let Some(r2) = n_left.get_register() {
                        Self::emit_mov_reg(ctx, r, r2);
                    }
                    let r = Self::bring_to_reg(ctx, right, vec![&l]);
                    (l, r)
                } else {
                    let r = if let Some(r) = ctx.allocate_register() {
                        r
                    } else {
                        let s = ctx.get_symbol_in_register(Some(vec![right])).unwrap();
                        Self::move_to_stack(&s, ctx);
                        ctx.allocate_register().unwrap()
                    };
                    let temp_name = ctx.get_temporary_name();
                    let l = ctx
                        .create_symbol(temp_name, LocalLocation::Register(r))
                        .unwrap();
                    Self::emit_mov_reg(ctx, r, right.get_register().unwrap());
                    (l, right.clone())
                }
            }
            LocalLocation::Stack(_) => {
                let r = if let Some(r) = ctx.allocate_register() {
                    r
                } else {
                    let purge = ctx.get_symbol_in_register(Some(vec![right])).unwrap();
                    let _ = Self::move_to_stack(&purge, ctx);
                    ctx.allocate_register().unwrap()
                };
                let temp_name = ctx.get_temporary_name();
                let l = ctx
                    .create_symbol(temp_name, LocalLocation::Register(r))
                    .unwrap();
                Self::emit_load_reg_reg(ctx, r, Reg::SP, left.get_stack_offset().unwrap());
                let r = Self::bring_to_reg(ctx, right, vec![&l]);
                (l, r)
            }
        }
    }

    fn compile_expr(
        expression: &Expression,
        ctx: &mut CompilerContext,
    ) -> Result<LocalInfo, String> {
        match expression {
            Expression::FunctionCall(_function_call) => todo!(),
            Expression::UnaryOperation(_unary_operation) => todo!(),
            Expression::BinaryOperation(binary_operation) => {
                let left = Self::compile_expr(&binary_operation.left, ctx)?;
                let right = Self::compile_expr(&binary_operation.right, ctx)?;

                // We have to pass an updated left as it could have moved when processing right
                let (left, right) = Self::adjust_for_binary_exp(
                    &ctx.find_local_info_by_name(left.get_name()).unwrap(),
                    &right,
                    ctx,
                );
                Self::emit_op(
                    ctx,
                    binary_operation.operator.value(),
                    left.get_register().unwrap(),
                    right.get_register().unwrap(),
                );
                if ctx.is_temporary(right.get_name()) {
                    let _ = ctx.destroy_symbol(&right);
                }
                return Ok(left);
            }
            Expression::Literal(literal) => return Self::compile_literal(literal, ctx),
        };
    }

    fn compile_fn_declaration(
        fd: &FunctionDeclaration,
        ctx: &mut CompilerContext,
    ) -> Result<(), String> {
        ctx.enter_function_declaration(fd);

        for stmt in &fd.body {
            match stmt {
                Statement::Expression(expression) => {
                    let v = Self::compile_expr(expression, ctx)?;
                    if ctx.is_temporary(v.get_name()) {
                        let _ = ctx.destroy_symbol(&v);
                    }
                }
                Statement::If(_) => todo!(),
                Statement::FunctionDeclaration(_function_declaration) => todo!(),
                Statement::Return(_) => todo!(),
                Statement::Let(_) => todo!(),
            }
        }

        ctx.exit_function_declaration();
        Ok(())
    }

    pub fn compile(&mut self) -> Result<Instructions, String> {
        let mut ctx = CompilerContext::new();
        ctx.build_function_temporary_location(&self.ast)?;

        for stmt in &self.ast {
            match stmt {
                Statement::FunctionDeclaration(function_declaration) => {
                    Self::compile_fn_declaration(function_declaration, &mut ctx)?;
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
